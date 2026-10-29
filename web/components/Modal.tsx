"use client";

import { useEffect, useRef } from "react";

interface Props {
  open: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
}

/** Accessible modal built on the native <dialog> element (focus trap + Esc to close for free). */
export default function Modal({ open, onClose, title, children }: Props) {
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);

  return (
    <dialog
      ref={ref}
      onClose={onClose}
      onClick={(e) => e.target === ref.current && onClose()}
      className="m-auto w-[min(28rem,calc(100vw-2rem))] rounded-xl p-0 shadow-xl backdrop:bg-slate-900/40"
    >
      <div className="space-y-4 p-5">
        <h2 className="text-lg font-semibold">{title}</h2>
        {children}
      </div>
    </dialog>
  );
}
