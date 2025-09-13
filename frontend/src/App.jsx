import React, { useState } from "react";
import ReactMarkdown from "react-markdown";

function App() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const res = await fetch("http://localhost:8000/legal-assistant", {  // use localhost instead of 127.0.0.1
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const data = await res.json();
    setResponse(data);
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Arial" }}>
      <h1>Legal Assistant</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter your legal query..."
          style={{ width: "400px", padding: "8px" }}
        />
        <button type="submit" style={{ marginLeft: "10px", padding: "8px 16px" }}>
          Search
        </button>
      </form>

      {response && (
        <div style={{ marginTop: "20px" }}>
          <h2>📌 IPC Analysis</h2>
          <ReactMarkdown>{response.ipc_analysis}</ReactMarkdown>
          <h2>⚖️ Case Precedents</h2>
          <ReactMarkdown>{response.precedents}</ReactMarkdown>
        </div>
      )}
    </div>
  );
}

export default App;
