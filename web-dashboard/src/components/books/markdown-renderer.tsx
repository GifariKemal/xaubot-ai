"use client";

import { Children, isValidElement, useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Components } from "react-markdown";
import { MermaidDiagram } from "./mermaid-diagram";

interface MarkdownRendererProps {
  content: string;
}

function getTextContent(node: React.ReactNode): string {
  if (typeof node === "string") return node;
  if (typeof node === "number") return String(node);
  if (!node) return "";
  if (Array.isArray(node)) return node.map(getTextContent).join("");
  if (isValidElement(node) && (node.props as Record<string, unknown>)?.children)
    return getTextContent((node.props as Record<string, unknown>).children as React.ReactNode);
  return "";
}

const components: Components = {
  // ── Headings ──
  h1: ({ children }) => (
    <h1 className="text-[2rem] font-bold text-slate-800 mt-10 mb-5 pb-3 border-b-2 border-blue-200/60 first:mt-0">
      {children}
    </h1>
  ),
  h2: ({ children }) => (
    <h2 className="text-[1.6rem] font-semibold text-slate-800 mt-9 mb-4 pb-2 border-b border-slate-200/80">
      {children}
    </h2>
  ),
  h3: ({ children }) => (
    <h3 className="text-[1.35rem] font-semibold text-slate-700 mt-6 mb-3 pl-3 border-l-3 border-blue-400/40">
      {children}
    </h3>
  ),
  h4: ({ children }) => (
    <h4 className="text-[1.15rem] font-semibold text-slate-700 mt-5 mb-2">
      {children}
    </h4>
  ),

  // ── Paragraphs & text ──
  p: ({ children }) => (
    <p className="text-[1.05rem] leading-[1.8] text-slate-600 mb-4">
      {children}
    </p>
  ),
  strong: ({ children }) => (
    <strong className="font-semibold text-slate-800">{children}</strong>
  ),
  em: ({ children }) => (
    <em className="italic text-slate-500">{children}</em>
  ),

  // ── Lists ──
  ul: ({ children }) => (
    <ul className="list-disc list-outside ml-5 mb-4 space-y-1.5 text-[1.05rem] text-slate-600">
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol className="list-decimal list-outside ml-5 mb-4 space-y-1.5 text-[1.05rem] text-slate-600">
      {children}
    </ol>
  ),
  li: ({ children }) => <li className="leading-[1.7] pl-1">{children}</li>,

  // ── Links ──
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-blue-600 hover:text-blue-800 underline underline-offset-2 decoration-blue-300 hover:decoration-blue-500 transition-colors"
    >
      {children}
    </a>
  ),

  // ── Blockquotes ──
  blockquote: ({ children }) => (
    <blockquote className="border-l-[3px] border-amber-400/60 pl-4 py-2 my-4 bg-amber-50/40 rounded-r-lg">
      <div className="text-slate-600 [&>p]:text-slate-600 [&>p]:mb-1">{children}</div>
    </blockquote>
  ),

  // ── Code blocks — with Mermaid detection ──
  pre: ({ children }) => {
    const child = Children.only(children);
    const childProps = isValidElement(child) ? (child.props as Record<string, unknown>) : null;
    if (childProps?.className === "language-mermaid") {
      const chart = getTextContent(childProps.children as React.ReactNode);
      return <MermaidDiagram chart={chart} />;
    }
    return (
      <pre className="my-4 p-4 rounded-xl bg-slate-50/80 border border-slate-200/80 overflow-x-auto text-[0.95rem] leading-relaxed">
        {children}
      </pre>
    );
  },
  code: ({ className, children }) => {
    const isBlock = className?.startsWith("language-");
    if (isBlock) {
      return (
        <code className="block font-mono text-slate-700">{children}</code>
      );
    }
    return (
      <code className="px-1.5 py-0.5 rounded-md bg-blue-50/70 text-blue-700 font-mono text-[0.95rem] border border-blue-100/50">
        {children}
      </code>
    );
  },

  // ── Tables — Excel-style ──
  table: ({ children }) => (
    <div className="my-5 overflow-x-auto rounded-xl border border-slate-200 shadow-sm">
      <table className="w-full text-[1rem] border-collapse">{children}</table>
    </div>
  ),
  thead: ({ children }) => (
    <thead className="bg-gradient-to-b from-slate-100 to-slate-50 border-b-2 border-slate-200">
      {children}
    </thead>
  ),
  th: ({ children }) => (
    <th className="px-4 py-2.5 text-left font-semibold text-slate-700 text-[0.95rem] uppercase tracking-wide border-r border-slate-200/60 last:border-r-0">
      {children}
    </th>
  ),
  tbody: ({ children }) => <tbody className="divide-y divide-slate-100">{children}</tbody>,
  tr: ({ children }) => (
    <tr className="hover:bg-blue-50/30 transition-colors even:bg-slate-50/40">
      {children}
    </tr>
  ),
  td: ({ children }) => (
    <td className="px-4 py-2.5 text-slate-600 border-r border-slate-100/80 last:border-r-0 align-top">
      {children}
    </td>
  ),

  // ── Horizontal rule ──
  hr: () => (
    <hr className="my-8 border-0 h-px bg-gradient-to-r from-transparent via-slate-300 to-transparent" />
  ),

  // ── Images (placeholder) ──
  img: ({ alt }) => (
    <span className="block my-4 text-center text-slate-400 text-sm italic">
      [{alt || "image"}]
    </span>
  ),
};

export function MarkdownRenderer({ content }: MarkdownRendererProps) {
  const processedContent = useMemo(() => content, [content]);

  return (
    <div className="max-w-none prose-slate">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {processedContent}
      </ReactMarkdown>
    </div>
  );
}
