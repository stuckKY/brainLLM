import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        mono: ['"Berkeley Mono"', '"SF Mono"', '"Fira Code"', "monospace"],
      },
      typography: {
        DEFAULT: {
          css: {
            fontFamily: '"Berkeley Mono", "SF Mono", "Fira Code", monospace',
            code: {
              fontFamily: '"Berkeley Mono", "SF Mono", "Fira Code", monospace',
            },
          },
        },
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};

export default config;
