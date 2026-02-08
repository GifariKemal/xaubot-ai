"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { usePrevious } from "./use-previous";

interface AnimatedValueResult {
  displayValue: number;
  direction: "up" | "down" | null;
  changeKey: number;
}

/**
 * Smoothly lerps a numeric value over `duration` ms using requestAnimationFrame.
 * Returns direction ("up"/"down"/null) and a changeKey that increments on each
 * value change — use it as a React `key` to replay CSS flash animations.
 */
export function useAnimatedValue(value: number, duration = 300): AnimatedValueResult {
  const prev = usePrevious(value);
  const [displayValue, setDisplayValue] = useState(value);
  const [direction, setDirection] = useState<"up" | "down" | null>(null);
  const [changeKey, setChangeKey] = useState(0);

  const rafRef = useRef<number>(0);
  const startRef = useRef(0);
  const fromRef = useRef(value);

  const animate = useCallback(
    (from: number, to: number) => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      fromRef.current = from;
      startRef.current = performance.now();

      const step = (now: number) => {
        const elapsed = now - startRef.current;
        const t = Math.min(elapsed / duration, 1);
        // ease-out cubic
        const eased = 1 - Math.pow(1 - t, 3);
        const current = fromRef.current + (to - fromRef.current) * eased;
        setDisplayValue(current);
        if (t < 1) {
          rafRef.current = requestAnimationFrame(step);
        }
      };

      rafRef.current = requestAnimationFrame(step);
    },
    [duration]
  );

  useEffect(() => {
    if (prev !== undefined && prev !== value) {
      setDirection(value > prev ? "up" : "down");
      setChangeKey((k) => k + 1);
      animate(displayValue, value);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  // Clear direction after flash duration
  useEffect(() => {
    if (direction === null) return;
    const timer = setTimeout(() => setDirection(null), 600);
    return () => clearTimeout(timer);
  }, [direction, changeKey]);

  // Cleanup raf on unmount
  useEffect(() => {
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  return { displayValue, direction, changeKey };
}
