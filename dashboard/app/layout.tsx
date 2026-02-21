// ============================================================
// layout.tsx - Root layout (wraps every page in the app)
// ============================================================
// This file defines the HTML skeleton and the sidebar that
// appears on every page. Next.js automatically wraps each
// page component inside this layout.

import type { Metadata } from "next";
import "./globals.css";
import ClientLayout from "@/components/ClientLayout";

// Metadata shown in the browser tab
export const metadata: Metadata = {
  title: "Crypto Trading Agent v7.0",
  description: "Real-time dashboard for the autonomous trading agent",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-gray-950 text-gray-100 antialiased">
        <ClientLayout>{children}</ClientLayout>
      </body>
    </html>
  );
}
