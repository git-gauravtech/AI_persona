const fs = require("fs");

const apiUrl = process.env.API_URL || "http://localhost:8000/chat";

const content = `
window.APP_CONFIG = {
  API_URL: "${apiUrl}"
};
`;

fs.writeFileSync("config.js", content.trim());
console.log("Generated config.js with API_URL:", apiUrl);