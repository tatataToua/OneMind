/**
 * Inline SVG icons, hand-rolled rather than pulled from a package.
 *
 * Two reasons, both of which this project has to be able to defend: the app
 * makes a point of running with no network calls, and an icon dependency is
 * the kind of thing that quietly adds a CDN font or a 200kB tree-shake
 * problem. Twelve paths cost less than either.
 *
 * All are 24x24, stroke 1.75, currentColor - so an icon inherits the colour of
 * whatever it sits in and needs no per-use styling.
 */

import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

function Svg({ size = 16, children, ...rest }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.75}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {children}
    </svg>
  );
}

export const IconPulse = (p: IconProps) => (
  <Svg {...p}>
    <path d="M3 12h4l2.5-7 5 14L17 12h4" />
  </Svg>
);

export const IconFlow = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="5" cy="6" r="2.4" />
    <circle cx="5" cy="18" r="2.4" />
    <circle cx="19" cy="12" r="2.4" />
    <path d="M7.4 6h3.1a2 2 0 0 1 2 2v2.1M7.4 18h3.1a2 2 0 0 0 2-2v-2.1M12.5 12h4.1" />
  </Svg>
);

export const IconShield = (p: IconProps) => (
  <Svg {...p}>
    <path d="M12 3 5 6v5.5c0 4 2.9 7.7 7 9.5 4.1-1.8 7-5.5 7-9.5V6z" />
    <path d="m9.2 12 2 2 3.6-3.8" />
  </Svg>
);

export const IconSend = (p: IconProps) => (
  <Svg {...p}>
    <path d="M12 19V5" />
    <path d="m6 11 6-6 6 6" />
  </Svg>
);

export const IconStop = (p: IconProps) => (
  <Svg {...p}>
    <rect x="6.5" y="6.5" width="11" height="11" rx="2" />
  </Svg>
);

export const IconAlert = (p: IconProps) => (
  <Svg {...p}>
    <path d="M12 4.5 2.8 20h18.4z" />
    <path d="M12 10v4" />
    <path d="M12 17.2h.01" />
  </Svg>
);

export const IconDatabase = (p: IconProps) => (
  <Svg {...p}>
    <ellipse cx="12" cy="6" rx="7" ry="2.8" />
    <path d="M5 6v12c0 1.5 3.1 2.8 7 2.8s7-1.3 7-2.8V6" />
    <path d="M5 12c0 1.5 3.1 2.8 7 2.8s7-1.3 7-2.8" />
  </Svg>
);

export const IconClock = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="12" cy="12" r="8.5" />
    <path d="M12 7.5V12l3 1.7" />
  </Svg>
);

export const IconChevron = (p: IconProps) => (
  <Svg {...p}>
    <path d="m9 5.5 6.5 6.5L9 18.5" />
  </Svg>
);

export const IconTag = (p: IconProps) => (
  <Svg {...p}>
    <path d="M11.6 3.5H4.5v7.1a2 2 0 0 0 .6 1.4l7.4 7.4a2 2 0 0 0 2.8 0l4.6-4.6a2 2 0 0 0 0-2.8L12.5 4.1a2 2 0 0 0-.9-.6z" />
    <path d="M8.3 8.3h.01" />
  </Svg>
);

export const IconCpu = (p: IconProps) => (
  <Svg {...p}>
    <rect x="6.5" y="6.5" width="11" height="11" rx="2" />
    <rect x="10" y="10" width="4" height="4" rx="1" />
    <path d="M9.5 3v3.5M14.5 3v3.5M9.5 17.5V21M14.5 17.5V21M3 9.5h3.5M3 14.5h3.5M17.5 9.5H21M17.5 14.5H21" />
  </Svg>
);

/**
 * The brand mark: a medical cross, drawn as two overlapping rounded bars.
 *
 * Original geometry rather than a traced logo, and filled rather than stroked -
 * everything else in this file is 1.75-weight line art sized for 14-16px, which
 * looks thin and incidental blown up to logotype size on the gradient tile.
 *
 * Two bars instead of one cross-shaped path so the arms can carry different
 * opacities: the vertical sits at 0.9 and the intersection reads a shade
 * brighter where they overlap, which gives the mark some depth without adding
 * a second colour it would have to coordinate with the tile behind it.
 *
 * Deliberately NOT a white cross on red. That specific combination is the Red
 * Cross emblem, protected by the Geneva Conventions and national law in most
 * countries - a genuine legal problem rather than a taste one, and an easy
 * thing to walk into when the brief is "make it look medical".
 */
export const IconMedicalCross = ({ size = 22, ...rest }: IconProps) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="currentColor"
    aria-hidden="true"
    focusable="false"
    {...rest}
  >
    <rect x="2.4" y="9.3" width="19.2" height="5.4" rx="2.5" opacity="0.9" />
    <rect x="9.3" y="2.4" width="5.4" height="19.2" rx="2.5" />
  </svg>
);
