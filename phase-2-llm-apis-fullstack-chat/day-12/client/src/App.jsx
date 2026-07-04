import { useState } from "react";
import ChatMessage from "./components/ChatMessage";
import MessageInput from "./components/MessageInput";

function App() {
  const [messages, setMessages] = useState([
    { role: "assistant", content: "Hi! Ask me anything." },
  ]);

  const [isLoading, setIsLoading] = useState(false);

  const handleSend = async (text) => {
    const userMessage = { role: "user", content: text };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });

      if (!response.ok) throw new Error("Server error");

      const data = await response.json();
      const assistantMessage = { role: "assistant", content: data.reply };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage = {
        role: "assistant",
        content: "Sorry, something went wrong. Please try again.",
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col">
      <header className="sticky top-0 bg-gray-800 border-b border-gray-700 px-6 py-4 z-10">
        <h1 className="text-white text-lg font-semibold max-w-2xl mx-auto">
          MindPilot Chat
        </h1>
      </header>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-2xl mx-auto">
          {messages.map((msg, index) => (
            <ChatMessage key={index} role={msg.role} content={msg.content} />
          ))}

          {isLoading && (
            <div className="flex justify-start mb-3">
              <div className="bg-gray-200 text-gray-500 px-4 py-2 rounded-2xl rounded-bl-sm text-sm">
                Typing...
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="max-w-2xl mx-auto w-full">
        <MessageInput onSend={handleSend} disabled={isLoading} />
      </div>
    </div>
  );
}

export default App;