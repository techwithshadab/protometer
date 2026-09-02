import type { ReactElement } from "react";

// Text wordmark in the BOTOX® house style (navy, uppercase, tight tracking). We render styled
// text rather than hotlinking the brand's logo asset. `suffix` appends e.g. "Assistant".
export function Wordmark({ suffix }: { suffix?: string }): ReactElement {
  return (
    <span className="wordmark">
      <span className="wordmark-name">
        BOTOX<sup className="wordmark-reg">®</sup>
      </span>
      {suffix ? <span className="wordmark-suffix">{suffix}</span> : null}
    </span>
  );
}
