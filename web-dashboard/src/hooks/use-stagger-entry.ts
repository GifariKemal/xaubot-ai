"use client";

import { useState, useEffect } from "react";

/**
 * Returns a boolean[] of length `count`. Each item becomes `true` after
 * `index * delayMs` milliseconds. Runs once on mount only.
 */
export function useStaggerEntry(count: number, delayMs = 40): boolean[] {
  const [visible, setVisible] = useState<boolean[]>(() => Array(count).fill(false));

  useEffect(() => {
    const timers: ReturnType<typeof setTimeout>[] = [];
    for (let i = 0; i < count; i++) {
      timers.push(
        setTimeout(() => {
          setVisible((prev) => {
            const next = [...prev];
            next[i] = true;
            return next;
          });
        }, i * delayMs)
      );
    }
    return () => timers.forEach(clearTimeout);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return visible;
}
