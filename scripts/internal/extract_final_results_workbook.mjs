import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";


const projectRoot = process.cwd();
const inputPath = path.join(
  projectRoot,
  "reports",
  "final_results",
  "final_performance_tables.xlsx",
);
const outputPath = path.join(
  projectRoot,
  ".artifact_tool_final_analysis",
  "final_performance_workbook_extract.json",
);

const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const mainMetrics = workbook.worksheets
  .getItem("Main Metrics")
  .getRange("A4:H9").values;
const mainInference = workbook.worksheets
  .getItem("Main Utility Inference")
  .getRange("A4:I9").values;

const errorScan = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final results workbook formula error scan",
});

await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.writeFile(
  outputPath,
  JSON.stringify(
    {
      source: inputPath,
      main_metrics: mainMetrics,
      main_utility_inference: mainInference,
      formula_error_scan: errorScan.ndjson,
    },
    null,
    2,
  ),
  "utf8",
);
console.log(`output=${outputPath}`);
