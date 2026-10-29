"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Library" },
  { href: "/search", label: "Search" },
];

export default function NavLinks() {
  const pathname = usePathname();
  return (
    <nav className="flex gap-1">
      {LINKS.map(({ href, label }) => {
        const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
        return (
          <Link
            key={href}
            href={href}
            className={`rounded-md px-3 py-1.5 text-sm font-medium ${
              active ? "bg-indigo-50 text-indigo-700" : "text-slate-600 hover:bg-slate-100"
            }`}
          >
            {label}
          </Link>
        );
      })}
    </nav>
  );
}
