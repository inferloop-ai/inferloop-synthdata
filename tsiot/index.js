require('dotenv').config();
const Anthropic = require('@anthropic-ai/sdk');

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

async function runClaude() {
  const message = await anthropic.messages.create({
    model: 'claude-3-opus-20240229',
    max_tokens: 500,
    messages: [{ role: 'user', content: 'What is synthetic data?' }],
  });

  console.log(message.content);
}

runClaude();

