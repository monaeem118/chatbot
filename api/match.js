const { createClient } = require("@supabase/supabase-js");
const { OpenAI } = require("openai");

// Initialize Supabase client
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Function to create embedding
async function createEmbedding(input) {
  try {
    const { data: embeddingData } = await openai.embeddings.create({
      model: "text-embedding-ada-002",
      input,
    });

    if (!embeddingData || !embeddingData[0]?.embedding) {
      throw new Error("Failed to generate embedding.");
    }

    return embeddingData[0].embedding;
  } catch (err) {
    console.error("Error in createEmbedding function:", err.message);
    throw err;
  }
}

// Function to find nearest match
async function findNearestMatch(embedding) {
  try {
    const { data, error } = await supabase.rpc("match_movies", {
      query_embedding: embedding,
      match_threshold: 0.5,
      match_count: 3,
    });

    if (error) {
      throw new Error(`Supabase query failed: ${error.message}`);
    }

    const match = data.map((obj) => obj.content).join("\n");
    return match;
  } catch (err) {
    console.error("Error in findNearestMatch function:", err.message);
    throw err;
  }
}

// System message setup
const chatMessages = [
  {
    role: "system",
    content: `You are an enthusiastic movie expert who loves recommending movies to people. 
    You will be given two pieces of information - some context about movies and a question. 
    Your main job is to formulate a short answer to the question using the provided context. 
    If you are unsure, say "Sorry, I don't know the answer." Do not make up answers.`,
  },
];

// Function to get chat completion
async function getChatCompletion(text, query) {
  try {
    chatMessages.push({
      role: "user",
      content: `Context: ${text} Question: ${query}`,
    });

    const { choices } = await openai.chat.completions.create({
      model: "gpt-4",
      messages: chatMessages,
      temperature: 0.65,
      frequency_penalty: 0.5,
    });

    const responseMessage = choices[0].message;
    chatMessages.push(responseMessage);
    console.log("Chat Response:", responseMessage.content);
    return responseMessage.content;
  } catch (error) {
    console.error("Error in getChatCompletion function:", error.message);
    throw new Error("Failed to generate a conversational response.");
  }
}

// Main logic function
async function main(input) {
  try {
    console.log("Thinking...");
    const embedding = await createEmbedding(input);
    const match = await findNearestMatch(embedding);
    const response = await getChatCompletion(match, input);
    console.log("Response:", response);
    return response;
  } catch (error) {
    console.error("Error in main function:", error.message);
    throw new Error("Sorry, something went wrong. Please try again.");
  }
}

// ====> The main exported handler for Vercel
module.exports = async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Only POST requests are allowed." });
  }

  const { input } = req.body;

  if (!input || typeof input !== "string") {
    return res
      .status(400)
      .json({ error: "Input is required and must be a string." });
  }

  try {
    const result = await main(input);

    if (!result) {
      return res.status(500).json({ error: "Failed to process input." });
    }

    return res.status(200).json({ recommendation: result });
  } catch (err) {
    console.error("Error in POST handler:", err.message);
    return res.status(500).json({ error: "Internal server error." });
  }
};
