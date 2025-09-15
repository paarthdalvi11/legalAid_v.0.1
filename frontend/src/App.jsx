import React, { useState } from "react";
import ReactMarkdown from "react-markdown";

function App() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState("");
  const [image, setImage] = useState(null);
  const [video, setVideo] = useState(null);
  const [loading, setLoading] = useState(false); // new

  const handleSearch = async () => {
    setResult("");      // clear previous result
    setLoading(true);   // show loading
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
      } else if (video) {
        const formData = new FormData();
        formData.append("file", video);

        const res = await fetch("http://localhost:8000/legal-assistant-video", {
          method: "POST",
          body: formData,
        });
        data = await res.json();
        setResult(`🎥 Video Analysis:\n\n${data.ipc_analysis}`);
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
    } finally {
      setLoading(false); // hide loading
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
        disabled={!!image || loading}
      />

      <input
        type="file"
        accept="image/*"
        onChange={(e) => setImage(e.target.files[0])}
        disabled={loading}
      />
      <input
        type="file"
        accept="video/*"
        onChange={(e) => setVideo(e.target.files[0])}
        disabled={loading}
      />

      <br />
      <button onClick={handleSearch} style={{ marginTop: "10px" }} disabled={loading}>
        Search
      </button>

      {loading && <p style={{ marginTop: "10px", fontWeight: "bold" }}>⏳ Processing... Please wait.</p>}

      <div style={{ marginTop: "20px" }}>
        <ReactMarkdown>{result}</ReactMarkdown>
      </div>
    </div>
  );
}

export default App;
