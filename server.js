const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');

require('dotenv').config();
const YANDEX_API_KEY = process.env.YANDEX_API_KEY;
const FOLDER_ID = process.env.FOLDER_ID;

const PORT = 3000;

// RAG: МЕДИЦИНСКАЯ БАЗА ЗНАНИЙ
const MEDICAL_DOCUMENTS = [
  "Клиническое руководство по фармакотерапии: Комбинация Магния (в т.ч. Магне B6) и Мелатонина является синергической. Магний способствует расслаблению мышечной ткани и модулирует ГАМК-рецепторы, что потенцирует седативный и снотворный эффект мелатонина. Совместный прием рекомендован при стресс-индуцированных расстройствах сна. Противопоказания: одновременный прием алкоголя, тяжелая почечная недостаточность.",
  "Взаимодействие этанола и мелатонина: Алкоголь значительно снижает эффективность мелатонина, нарушая фазы сна. Одновременный прием может вызвать дезориентацию, головную боль и повышенную нагрузку на печень. Сочетание строго противопоказано.",
  "Фармакодинамика кальция и магния: Кальций и магний конкурируют за одни и те же каналы всасывания в ЖКТ. Одновременный прием в высоких дозах снижает биодоступность магния. Рекомендуется разносить прием препаратов минимум на 2 часа.",
  "Инструкция по применению Супрастина: Супрастин (хлоропирамин) — антигистаминный препарат. Обладает выраженным седативным и снотворным действием. Категорически запрещено совмещать с алкоголем и другими депрессантами ЦНС."
];

let VECTOR_DATABASE = [];

function getEmbedding(text, queryType = 'text-search-doc') {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify({
      modelUri: `emb://${FOLDER_ID}/${queryType}/latest`,
      text: text
    });
    const options = {
      hostname: 'llm.api.cloud.yandex.net',
      path: '/foundationModels/v1/textEmbedding',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Api-Key ${YANDEX_API_KEY}`,
        'x-folder-id': FOLDER_ID,
        'Content-Length': Buffer.byteLength(body)
      }
    };
    const req = https.request(options, res => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          const json = JSON.parse(data);
          if (json.embedding) resolve(json.embedding);
          else reject(new Error("Нет эмбеддинга: " + data));
        } catch (e) { reject(e); }
      });
    });
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

function cosineSimilarity(vecA, vecB) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dot += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function initVectorDatabase() {
  console.log("⏳ [RAG] Векторизация медицинских документов...");
  try {
    for (const doc of MEDICAL_DOCUMENTS) {
      const vector = await getEmbedding(doc, 'text-search-doc');
      VECTOR_DATABASE.push({ text: doc, vector });
    }
    console.log(`✅ [RAG] База готова! Документов: ${VECTOR_DATABASE.length}`);
  } catch (error) {
    console.error("❌ [RAG] Ошибка инициализации:", error.message);
  }
}

async function queryVectorDatabase(userText) {
  try {
    const userVector = await getEmbedding(userText, 'text-search-query');
    let bestMatch = null, maxSim = -1;
    for (const item of VECTOR_DATABASE) {
      const sim = cosineSimilarity(userVector, item.vector);
      if (sim > maxSim) { maxSim = sim; bestMatch = item; }
    }
    if (maxSim > 0.35) {
      console.log(`🎯 [RAG] Совпадение: ${(maxSim * 100).toFixed(1)}%`);
      return `[ВЕРИФИЦИРОВАННЫЙ ДОКУМЕНТ]: ${bestMatch.text}\n`;
    }
  } catch (e) {
    console.error("❌ [RAG] Ошибка поиска:", e.message);
  }
  return "";
}
function tryRecognize(audioBuffer, formats, index, res) {
  if (index >= formats.length) {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error_message: 'Не удалось распознать ни в одном формате. Попробуйте MP3.' }));
    return;
  }

  const fmt = formats[index];
  const sampleRate = fmt === 'lpcm' ? '&sampleRateHertz=16000' : '';
  const sttPath = `/speech/v1/stt:recognize?folderId=${FOLDER_ID}&lang=ru-RU&format=${fmt}${sampleRate}`;

  console.log(`🎙️ [SpeechKit] Пробую формат: ${fmt}`);

  const options = {
    hostname: 'stt.api.cloud.yandex.net',
    path: sttPath,
    method: 'POST',
    headers: {
      'Authorization': `Api-Key ${YANDEX_API_KEY}`,
      'Content-Type': 'application/octet-stream',
      'Content-Length': audioBuffer.length
    },
    timeout: 30000
  };

  const sttReq = https.request(options, sttRes => {
    let data = '';
    sttRes.on('data', chunk => data += chunk);
    sttRes.on('end', () => {
      console.log(`🎙️ [SpeechKit] Статус (${fmt}):`, sttRes.statusCode, data);
      try {
        const parsed = JSON.parse(data);
        if (parsed.result) {
          // Успех
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(data);
        } else {
          // Пробуем следующий формат
          console.log(`🎙️ [SpeechKit] Формат ${fmt} не подошёл, пробую следующий...`);
          tryRecognize(audioBuffer, formats, index + 1, res);
        }
      } catch(e) {
        tryRecognize(audioBuffer, formats, index + 1, res);
      }
    });
  });

  sttReq.on('error', err => {
    console.error(`❌ [SpeechKit] Ошибка (${fmt}):`, err.message);
    // При ошибке соединения пробуем следующий формат
    tryRecognize(audioBuffer, formats, index + 1, res);
  });

  sttReq.on('timeout', () => {
    console.error(`❌ [SpeechKit] Таймаут (${fmt})`);
    sttReq.destroy();
    tryRecognize(audioBuffer, formats, index + 1, res);
  });

  sttReq.write(audioBuffer);
  sttReq.end();
}
// =========================================================================
// HTTP СЕРВЕР
// =========================================================================
const server = http.createServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') { res.writeHead(200); res.end(); return; }

  // -----------------------------------------------------------------------
  // МАРШРУТ 1: YandexGPT — чат с Алисой
  // -----------------------------------------------------------------------
  if (req.method === 'POST' && req.url === '/api/alice') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', async () => {
      let parsedBody;
      try { parsedBody = JSON.parse(body); } catch(e) { parsedBody = {}; }

      let incomingMessages = parsedBody.messages || [];
      const lastUserMessage = incomingMessages.filter(m => m.role === 'user').pop()?.text || "";
      const medicalContext = lastUserMessage ? await queryVectorDatabase(lastUserMessage) : "";

      if (medicalContext) {
        const instruction = `\n\nПримени этот проверенный медицинский контекст:\n${medicalContext}`;
        if (incomingMessages.length > 0 && incomingMessages[0].role === 'system') {
          incomingMessages[0].text += instruction;
        } else {
          incomingMessages.unshift({ role: 'system', text: instruction });
        }
      }

      const yandexBody = JSON.stringify({
        modelUri: `gpt://${FOLDER_ID}/yandexgpt/latest`,
        completionOptions: { stream: false, temperature: 0.1, maxTokens: 1000 },
        messages: incomingMessages
      });

      const options = {
        hostname: 'llm.api.cloud.yandex.net',
        path: '/foundationModels/v1/completion',
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Api-Key ${YANDEX_API_KEY}`,
          'x-folder-id': FOLDER_ID,
          'Content-Length': Buffer.byteLength(yandexBody)
        }
      };

      const proxyReq = https.request(options, proxyRes => {
        let data = '';
        proxyRes.on('data', chunk => data += chunk);
        proxyRes.on('end', () => {
          res.writeHead(proxyRes.statusCode, { 'Content-Type': 'application/json' });
          res.end(data);
        });
      });
      proxyReq.on('error', err => { res.writeHead(500); res.end(JSON.stringify({ error: err.message })); });
      proxyReq.write(yandexBody);
      proxyReq.end();
    });
    return;
  }

  /// -----------------------------------------------------------------------
  // МАРШРУТ 2: SpeechKit — распознавание голоса
  // -----------------------------------------------------------------------
  if (req.method === 'POST' && req.url.startsWith('/api/speechkit')) {
    const urlParams = new URL('http://localhost' + req.url).searchParams;
    const format = urlParams.get('format') || 'mp3';

    console.log(`🎙️ [SpeechKit] Формат: ${format}`);

    let chunks = [];
    req.on('data', chunk => chunks.push(chunk));
    req.on('end', () => {
      const audioBuffer = Buffer.concat(chunks);
      console.log(`🎙️ [SpeechKit] Размер: ${audioBuffer.length} байт`);

      // OGG из Telegram бывает как opus так и vorbis — пробуем оба
      const formatsToTry = format === 'oggopus'
        ? ['oggopus', 'mp3']
        : [format];

      tryRecognize(audioBuffer, formatsToTry, 0, res);
    });
    return;
  }

  // -----------------------------------------------------------------------
  // МАРШРУТ 3: Отдача HTML файла
  // -----------------------------------------------------------------------
  if (req.method === 'GET' && (req.url === '/' || req.url === '/index.html' || req.url === '/alica-health-ai.html')) {
    const filePath = path.join(__dirname, 'alica-health-ai.html');
    fs.readFile(filePath, (err, data) => {
      if (err) { res.writeHead(404); res.end('Файл не найден'); return; }
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end(data);
    });
    return;
  }

  res.writeHead(404);
  res.end('Not found');
});

initVectorDatabase().then(() => {
  server.listen(PORT, () => {
    console.log(`🚀 Сервер запущен: http://localhost:${PORT}`);
    console.log(`📡 Эндпоинты: /api/alice · /api/speechkit`);
  });
});