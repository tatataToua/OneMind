/**
 * The smallest markdown subset the specialists actually emit.
 *
 * Deliberately not a markdown library. It handles `**bold**`, `` `code` ``,
 * and `-`/`*` bullet lists, and treats everything else as plain text. Two
 * reasons: the answer is written by a language model, so building React nodes
 * instead of an HTML string removes the injection surface entirely; and a
 * clinical reader should never see a construct the renderer got half right.
 *
 * The Developer tab prints the same string unparsed - there, the literal bytes
 * are the point.
 */

import type { ReactNode } from "react";

const INLINE = /(\*\*[^*\n]+\*\*|`[^`\n]+`)/g;
const BULLET = /^\s*[-*]\s+(.*)$/;

function inline(text: string, keyBase: string): ReactNode[] {
  return text.split(INLINE).map((part, i) => {
    const key = keyBase + ":" + i;
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return <strong key={key}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return <code key={key}>{part.slice(1, -1)}</code>;
    }
    return part;
  });
}

export function Markdown({ text }: { text: string }) {
  const lines = text.split("\n");
  const blocks: ReactNode[] = [];

  // Consecutive bullet lines become one list; consecutive prose lines become
  // one paragraph. A blank line closes whatever is open.
  let bullets: string[] = [];
  let para: string[] = [];

  const flushBullets = () => {
    if (bullets.length === 0) return;
    const items = bullets;
    bullets = [];
    blocks.push(
      <ul key={"ul" + blocks.length}>
        {items.map((item, i) => (
          <li key={i}>{inline(item, "b" + blocks.length + "-" + i)}</li>
        ))}
      </ul>,
    );
  };

  const flushPara = () => {
    if (para.length === 0) return;
    const body = para.join("\n");
    para = [];
    blocks.push(<p key={"p" + blocks.length}>{inline(body, "p" + blocks.length)}</p>);
  };

  for (const line of lines) {
    const bullet = line.match(BULLET);
    if (bullet) {
      flushPara();
      bullets.push(bullet[1]);
    } else if (line.trim() === "") {
      flushBullets();
      flushPara();
    } else {
      flushBullets();
      para.push(line);
    }
  }
  flushBullets();
  flushPara();

  return <>{blocks}</>;
}
