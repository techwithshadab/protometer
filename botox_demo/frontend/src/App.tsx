import { useState } from "react";
import { ChatWidget } from "./components/ChatWidget";
import { LandingPage } from "./components/landing/LandingPage";

// The app renders the demo landing page as the backdrop, with the floating chat widget on top.
// ChatWidget manages its own open/closed state via these props; the LandingPage is purely visual
// and sits below the widget's fixed, high-z-index layer.
export function App() {
  const [open, setOpen] = useState(false);
  return (
    <>
      <LandingPage />
      <ChatWidget open={open} onOpenChange={setOpen} />
    </>
  );
}
