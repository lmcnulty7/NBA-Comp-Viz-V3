import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

// Attributes whose value is read or heard by a person. Their text is copy, so it lives in web/content.
const TEXT_ATTRIBUTE = "/^(aria-label|aria-description|aria-roledescription|aria-valuetext|aria-placeholder|alt|title|placeholder|label|summary)$/";
const COPY_MESSAGE =
  "Site copy lives in web/content and reaches a component through lib/copy.ts. A number comes from the data root through lib/format.ts.";
const DATA_MESSAGE = "Pages and components get data only through a DataSource (lib/data/source.ts).";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    // Generated or copied files:
    "lib/types/contract.ts",
    "public/**",
    "data/**",
    "test-results/**",
    "playwright-report/**",
  ]),
  {
    // No copy and no number typed into a page or a component (PLAN.md: no numeric literals in copy).
    files: ["app/**/*.{ts,tsx}", "components/**/*.{ts,tsx}"],
    rules: {
      // Literal text between tags, as a string, or as a template. Lone punctuation is allowed.
      "react/jsx-no-literals": [
        "error",
        { noStrings: true, ignoreProps: true, allowedStrings: [".", ",", ":", "/", "(", ")"] },
      ],
      "no-restricted-syntax": [
        "error",
        // Text hidden in one branch of a condition between tags.
        { selector: "JSXElement > JSXExpressionContainer > ConditionalExpression > Literal[value=/[A-Za-z0-9]/]", message: COPY_MESSAGE },
        { selector: "JSXFragment > JSXExpressionContainer > ConditionalExpression > Literal[value=/[A-Za-z0-9]/]", message: COPY_MESSAGE },
        { selector: "JSXElement > JSXExpressionContainer > ConditionalExpression > ConditionalExpression > Literal[value=/[A-Za-z0-9]/]", message: COPY_MESSAGE },
        { selector: "JSXElement > JSXExpressionContainer > LogicalExpression > Literal[value=/[A-Za-z0-9]/]", message: COPY_MESSAGE },
        { selector: "JSXFragment > JSXExpressionContainer > LogicalExpression > Literal[value=/[A-Za-z0-9]/]", message: COPY_MESSAGE },
        { selector: "JSXElement > JSXExpressionContainer > ConditionalExpression > TemplateLiteral", message: COPY_MESSAGE },
        { selector: "JSXElement > JSXExpressionContainer > LogicalExpression > TemplateLiteral", message: COPY_MESSAGE },
        // A number typed between tags.
        { selector: "JSXElement > JSXExpressionContainer > Literal[raw=/^[0-9]/]", message: COPY_MESSAGE },
        { selector: "JSXFragment > JSXExpressionContainer > Literal[raw=/^[0-9]/]", message: COPY_MESSAGE },
        // Text typed into an attribute a person reads or hears.
        { selector: `JSXAttribute[name.name=${TEXT_ATTRIBUTE}] > Literal`, message: COPY_MESSAGE },
        { selector: `JSXAttribute[name.name=${TEXT_ATTRIBUTE}] > JSXExpressionContainer > Literal`, message: COPY_MESSAGE },
        { selector: `JSXAttribute[name.name=${TEXT_ATTRIBUTE}] > JSXExpressionContainer > TemplateLiteral`, message: COPY_MESSAGE },
        { selector: `JSXAttribute[name.name=${TEXT_ATTRIBUTE}] > JSXExpressionContainer > ConditionalExpression > Literal[value=/[A-Za-z0-9]/]`, message: COPY_MESSAGE },
        { selector: `JSXAttribute[name.name=${TEXT_ATTRIBUTE}] > JSXExpressionContainer > ConditionalExpression > TemplateLiteral`, message: COPY_MESSAGE },
      ],
      // Data comes through the DataSource: no file reads and no fetch in a page or a component.
      "no-restricted-imports": [
        "error",
        {
          paths: ["fs", "fs/promises", "node:fs", "node:fs/promises", "path", "node:path"].map((name) => ({
            name,
            message: DATA_MESSAGE,
          })),
        },
      ],
      "no-restricted-globals": [
        "error",
        { name: "fetch", message: DATA_MESSAGE },
        { name: "XMLHttpRequest", message: DATA_MESSAGE },
      ],
      "no-restricted-properties": [
        "error",
        { object: "window", property: "fetch", message: DATA_MESSAGE },
        { object: "globalThis", property: "fetch", message: DATA_MESSAGE },
      ],
    },
  },
]);

export default eslintConfig;
