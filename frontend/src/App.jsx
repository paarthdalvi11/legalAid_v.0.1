import React, { useState } from "react";
import ReactMarkdown from "react-markdown";

function App() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState("");
  const [image, setImage] = useState(null);

  const handleSearch = async () => {
    try {
      let data;
      if (image) {
        const formData = new FormData();
        formData.append("file", image);

        const res = await fetch("http://localhost:8000/legal-assistant-image", {
          method: "POST",
          body: formData,
        });
        data = await res.json();
        setResult(`📌 Caption: ${data.caption}\n\n${data.ipc_analysis}`);
      } else {
        const res = await fetch("http://localhost:8000/legal-assistant", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query }),
        });
        data = await res.json();
        setResult(data.ipc_analysis);
      }
    } catch (err) {
      console.error(err);
      setResult("❌ Error: Could not fetch response from backend.");
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>⚖️ Legal Assistant</h1>

      <textarea
        placeholder="Enter your query..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        style={{ width: "100%", height: "80px", marginBottom: "10px" }}
        disabled={!!image}
      />

      <input
        type="file"
        accept="image/*"
        onChange={(e) => setImage(e.target.files[0])}
      />

      <br />
      <button onClick={handleSearch} style={{ marginTop: "10px" }}>
        Search
      </button>

      <div style={{ marginTop: "20px" }}>
        <ReactMarkdown>{result}</ReactMarkdown>
      </div>
    </div>
  );
}

// HAHAHAHAHA TU TOH GAYA BETAAA!!


export default App;
