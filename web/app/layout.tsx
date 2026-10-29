import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import NavLinks from "@/components/NavLinks";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "VisionFind",
  description: "Find and understand moments inside your videos using plain language.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      {/* Browser extensions (e.g. Grammarly) add attributes to <body> before hydration. */}
      <body className={`${geistSans.variable} ${geistMono.variable} font-sans`} suppressHydrationWarning>
        <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/90 backdrop-blur">
          <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4">
            <Link href="/" className="text-lg font-semibold tracking-tight">
              Vision<span className="text-indigo-600">Find</span>
            </Link>
            <NavLinks />
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-4 py-6">{children}</main>
      </body>
    </html>
  );
}
