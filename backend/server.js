// backend/server.js
// Full upgraded live-news backend for Fake News Detector
// Dependencies (install in backend/): express cors body-parser dotenv helmet express-rate-limit node-cache express-validator ajv groq-sdk node-fetch
// npm install express cors body-parser dotenv helmet express-rate-limit node-cache express-validator ajv groq-sdk node-fetch

const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
require('dotenv').config();

const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const NodeCache = require('node-cache');
const { body, validationResult } = require('express-validator');
const Ajv = require('ajv');
const fs = require('fs');
const path = require('path');

// Groq SDK
const Groq = require('groq-sdk');

// Resilient fetch: prefer global fetch (Node 18+), then try node-fetch (v2 or v3), else null
let fetchFn = (typeof fetch !== 'undefined') ? fetch : null;
if (!fetchFn) {
  try {
    const nf = require('node-fetch');
    // node-fetch v3 is ESM in some installs and exposes default; this handles both
    fetchFn = nf && (nf.default || nf);
  } catch (e) {
    console.warn('node-fetch not installed or failed to load. Install node-fetch@2 for CommonJS compatibility:', e && e.message ? e.message : e);
    fetchFn = null;
  }
}

const app = express();
app.use(helmet());
app.use(cors());
app.use(bodyParser.json({ limit: '300kb' }));

// =========================
// Config & SDK setup
// =========================
const GROQ_API_KEY = process.env.GROQ_API_KEY || '';
const GROQ_MODEL = process.env.GROQ_MODEL || 'llama-3.3-70b-versatile';
const NEWSAPI_KEY = process.env.NEWSAPI_KEY || '';
const CLIENT_API_KEYS = (process.env.CLIENT_API_KEYS || '').split(',').map(s => s.trim()).filter(Boolean);
const PORT = process.env.PORT || 3000;
const FRONTEND_URL = process.env.FRONTEND_URL || '';

if (!GROQ_API_KEY) console.warn('⚠️ GROQ_API_KEY not set in .env — model calls will fail without it');
if (!NEWSAPI_KEY) console.warn('ℹ️ NEWSAPI_KEY not set — evidence lookup will be disabled (model will fallback to general knowledge)');

const groq = new Groq({ apiKey: GROQ_API_KEY });

// =========================
// Serve static frontend if present
// =========================
const frontendPath = path.join(__dirname, '..', 'frontend');
if (fs.existsSync(path.join(frontendPath, 'index.html'))) {
  app.use(express.static(frontendPath));
  // for SPA fallback
  // SPA fallback - serve index.html for any non-/api route
// Use a regex route to avoid path-to-regexp '*' parsing errors.
app.get(/^\/(?!api).*/, (req, res) => {
  res.sendFile(path.join(frontendPath, 'index.html'));
});

}

// =========================
// Utilities: cache, files
// =========================
const cache = new NodeCache({ stdTTL: 600, checkperiod: 120 }); // default 10 minutes TTL

const REVIEWS_FILE = path.join(__dirname, 'reviews.json');
function ensureReviewsFile() {
  try {
    if (!fs.existsSync(REVIEWS_FILE)) {
      fs.writeFileSync(REVIEWS_FILE, JSON.stringify([], null, 2), { encoding: 'utf8' });
      console.log('Created reviews.json');
    }
  } catch (e) {
    console.error('Failed to ensure reviews.json:', e);
  }
}
ensureReviewsFile();

function readReviews() {
  try {
    const raw = fs.readFileSync(REVIEWS_FILE, 'utf8');
    return JSON.parse(raw || '[]');
  } catch (e) {
    console.error('Failed to read reviews.json:', e);
    return [];
  }
}
function writeReviews(revs) {
  try {
    fs.writeFileSync(REVIEWS_FILE, JSON.stringify(revs, null, 2), 'utf8');
  } catch (e) {
    console.error('Failed to write reviews.json:', e);
  }
}

// =========================
// Rate limiting & auth
// =========================
const apiLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 40,
  message: { error: 'Too many requests. Slow down.' }
});
app.use('/api/', apiLimiter);

function requireClientKey(req, res, next) {
  if (!CLIENT_API_KEYS.length) return next(); // dev mode: no key required
  const key = (req.headers['x-api-key'] || '').trim();
  if (!key) return res.status(401).json({ error: 'x-api-key header required' });
  if (!CLIENT_API_KEYS.includes(key)) return res.status(403).json({ error: 'Invalid API key' });
  return next();
}

// =========================
// Input validation
// =========================
const validateCheckInput = [
  body('text').isString().isLength({ min: 5, max: 5000 }).trim()
];

// =========================
// Schema for model output (AJV)
// =========================
const ajv = new Ajv({ allErrors: true, strict: false });
const outputSchema = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['REAL', 'FAKE', 'MIXED', 'UNSURE'] },
    confidence: { type: 'integer', minimum: 0, maximum: 100 },
    explanation: { type: 'string' },
    sources: { type: 'array', items: { type: 'string' } }
  },
  required: ['verdict', 'confidence', 'explanation', 'sources'],
  additionalProperties: false
};
const validateOutput = ajv.compile(outputSchema);

// =========================
// Helper: Fetch news snippets (NewsAPI)
// =========================
async function fetchNewsSnippetsCached(query) {
  const crypto = require('crypto');
  const cacheKey = 'news:' + crypto.createHash('sha256').update(query).digest('hex');
  const cached = cache.get(cacheKey);
  if (cached) return cached;

  if (!NEWSAPI_KEY || !fetchFn) {
    cache.set(cacheKey, [] , 60); // cache empty for short time
    return [];
  }

  try {
    const q = encodeURIComponent(query);
    const url = `https://newsapi.org/v2/everything?q=${q}&language=en&pageSize=5&sortBy=publishedAt&apiKey=${NEWSAPI_KEY}`;
    const r = await fetchFn(url, { timeout: 8000 });
    if (!r || !r.ok) {
      console.warn('NewsAPI response not ok', r && r.status);
      cache.set(cacheKey, [], 60);
      return [];
    }
    const j = await r.json();
    const articles = (j.articles || []).map(a => ({
      title: a.title || '',
      source: (a.source && a.source.name) ? a.source.name : '',
      publishedAt: a.publishedAt || '',
      snippet: (a.description || '').replace(/\s+/g, ' ').trim(),
      url: a.url || ''
    }));
    cache.set(cacheKey, articles, 300); // 5 minutes
    return articles;
  } catch (e) {
    console.error('News fetch error:', e && e.message ? e.message : e);
    cache.set(cacheKey, [], 60);
    return [];
  }
}

// =========================
// Health endpoint
// =========================
app.get('/api/health', (req, res) => {
  res.json({ ok: true, time: new Date().toISOString(), env: { hasNewsKey: Boolean(NEWSAPI_KEY), hasGroqKey: Boolean(GROQ_API_KEY) } });
});

// =========================
// Main fact-check endpoint
// =========================
app.post('/api/check', requireClientKey, validateCheckInput, async (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) return res.status(400).json({ error: 'Invalid input', details: errors.array() });

  const userText = String(req.body.text || '').trim();
  if (!userText) return res.status(400).json({ error: 'Empty text' });

  // cache check
  const crypto = require('crypto');
  const cacheKey = crypto.createHash('sha256').update(userText).digest('hex');
  const cached = cache.get(cacheKey);
  if (cached) return res.json({ ...cached, cached: true });

  // Fetch news snippets (if available)
  const snippets = await fetchNewsSnippetsCached(userText);
  let searchResultsText = 'NONE';
  if (snippets && snippets.length) {
    searchResultsText = snippets.map((s, i) =>
      `${i+1}) [${s.source}] ${s.title} (${s.publishedAt || 'unknown'})\n${s.snippet}\n${s.url}`
    ).join('\n\n');
  }

  const prompt = `
You are a precise fact-checker. FOLLOW THESE RULES:

1) Output ONLY valid JSON and NOTHING else.
2) Use EXACT JSON schema:
{
  "verdict": "REAL|FAKE|MIXED|UNSURE",
  "confidence": number,
  "explanation": "short text",
  "sources": ["..."]
}

SEARCH_RESULTS:
${searchResultsText}

Now analyze the Text below and output exactly one JSON object that follows the schema above. Keep explanation short.
"""${userText}"""
`;

  try {
    // Call Groq
    const response = await groq.chat.completions.create({
      model: GROQ_MODEL,
      temperature: 0,
      max_tokens: 900,
      messages: [
        { role: 'system', content: 'You are a careful fact checker.' },
        { role: 'user', content: prompt }
      ]
    });

    const raw = (response?.choices?.[0]?.message?.content || '').trim();
    console.log('--- RAW MODEL OUTPUT START ---');
    console.log(raw);
    console.log('--- RAW MODEL OUTPUT END ---');

    // Try strict JSON parse then fallback to first {...} block
    let parsed = null;
    try { parsed = JSON.parse(raw); } catch (e) { parsed = null; }

    if (!parsed) {
      // best-effort extract first balanced {...}
      const text = raw || '';
      const start = text.indexOf('{');
      if (start !== -1) {
        let depth = 0, end = -1;
        for (let i = start; i < text.length; i++) {
          if (text[i] === '{') depth++;
          else if (text[i] === '}') depth--;
          if (depth === 0) { end = i; break; }
        }
        if (end !== -1) {
          try { parsed = JSON.parse(text.slice(start, end + 1)); } catch (e2) { parsed = null; }
        }
      }
    }

    if (!parsed) {
      return res.status(502).json({ error: 'Invalid JSON from model', raw });
    }

    // Validate against schema
    if (!validateOutput(parsed)) {
      return res.status(502).json({ error: 'Model JSON does not match schema', validationErrors: validateOutput.errors, raw: parsed });
    }

    // If model did not include sources but snippets exist, add top snippet URLs as fallback
    if ((!parsed.sources || !parsed.sources.length) && snippets && snippets.length) {
      parsed.sources = (parsed.sources || []).concat(snippets.map(s => s.url).filter(Boolean));
      parsed.sources = Array.from(new Set(parsed.sources)).slice(0, 5);
    }

    // persist in cache & return
    cache.set(cacheKey, parsed);
    return res.json(parsed);

  } catch (err) {
    console.error('=== Groq Call Error ===', err?.message || err);
    if (err?.response?.data?.error?.code === 'model_decommissioned') {
      return res.status(502).json({
        error: 'Model decommissioned',
        message: err.response.data.error.message,
        help: 'Update GROQ_MODEL in .env to a supported model.'
      });
    }

    return res.status(500).json({ error: 'Server error', details: err?.response?.data || err?.message || String(err) });
  }
});

// =========================
// Reviews endpoints
// =========================
app.get('/api/reviews', (req, res) => {
  const limit = Math.min(100, parseInt(req.query.limit) || 40);
  const reviews = readReviews();
  return res.json({ reviews: reviews.slice().reverse().slice(0, limit) });
});

app.post('/api/reviews',
  [
    body('name').optional().isString().trim().isLength({ min: 1, max: 50 }).escape(),
    body('comment').isString().trim().isLength({ min: 3, max: 500 }).escape()
  ],
  (req, res) => {
    if (CLIENT_API_KEYS.length) {
      const key = (req.headers['x-api-key'] || '').trim();
      if (!key) return res.status(401).json({ error: 'x-api-key header required' });
      if (!CLIENT_API_KEYS.includes(key)) return res.status(403).json({ error: 'Invalid API key' });
    }

    console.log('Incoming review POST body:', req.body);

    const errors = validationResult(req);
    if (!errors.isEmpty()) return res.status(400).json({ error: 'Invalid input', details: errors.array() });

    const name = req.body.name ? String(req.body.name).trim() : 'Anonymous';
    const comment = String(req.body.comment).trim();

    const reviews = readReviews();
    const entry = {
      id: Date.now().toString(36) + Math.random().toString(36).slice(2, 8),
      name,
      comment,
      createdAt: new Date().toISOString()
    };
    reviews.push(entry);
    writeReviews(reviews);

    return res.status(201).json({ message: 'Review added', review: entry });
  }
);

// =========================
// Start server
// =========================
app.listen(PORT, () => {
  console.log(`Backend running on port ${PORT}`);
});
