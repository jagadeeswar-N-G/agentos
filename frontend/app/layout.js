import { IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

const ibmPlexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
});

export const metadata = {
  title: "AgentOS — Support Agent Demo",
  description: "Production multi-agent AI support system",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body style={{ fontFamily: ibmPlexMono.style.fontFamily, position: "relative", zIndex: 1 }}>
        {children}
      </body>
    </html>
  );
}