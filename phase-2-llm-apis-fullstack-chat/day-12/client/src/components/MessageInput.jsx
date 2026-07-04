import { useState } from "react";

function MessageInput({ onSend, disabled }) {
  const [text, setText] = useState("");

  const handleSend = () => {
    const trimmed = text.trim();
    if (!trimmed) return;
    onSend(trimmed);
    setText("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex items-end gap-2 border-t border-gray-700 p-3 bg-gray-900">
      <div className="flex-1">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
          placeholder="Type a message..."
          disabled={disabled}
          className="w-full resize-none rounded-xl bg-gray-800 text-white px-4 py-2
                     text-sm outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
        />
        <div className="text-xs text-gray-500 mt-1 text-right">
          {text.length} characters
        </div>
      </div>

      <button
        onClick={handleSend}
        disabled={disabled || !text.trim()}
        className="bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700
                   disabled:cursor-not-allowed text-white px-4 py-2 rounded-xl text-sm"
      >
        Send
      </button>
    </div>
  );
}

export default MessageInput;