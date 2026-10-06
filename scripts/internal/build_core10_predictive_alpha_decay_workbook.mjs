import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const outputRoot = path.join(
  projectRoot,
  "outputs",
  "extensions",
  "core10_predictive_alpha_decay",
);
const outputPath = path.join(
  outputRoot,
  "core10_predictive_alpha_decay_analysis.xlsx",
);
const previewRoot = path.join(
  os.tmpdir(),
  "core10_predictive_alpha_decay_workbook_previews",
);

const sheets = [
  ["README", "workbook_readme.csv", "A1:B18"],
  ["Timing_Audit", "timing_audit.csv", "A1:K13"],
  ["Horizon_Sample_Counts", "horizon_sample_counts.csv", "A1:L13"],
  ["Monthly_RankIC", "monthly_rank_ic.csv", "A1:H18"],
  ["RankIC_Summary", "rank_ic_summary.csv", "A1:J18"],
  ["Monthly_PearsonIC", "monthly_pearson_ic.csv", "A1:H18"],
  ["PearsonIC_Summary", "pearson_ic_summary.csv", "A1:J18"],
  ["Monthly_FMB_Slopes", "monthly_fmb_slopes.csv", "A1:I18"],
  ["FMB_Summary", "fmb_summary.csv", "A1:L18"],
  ["Alpha_Persistence_Metrics", "alpha_persistence_metrics.csv", "A1:L12"],
  ["Characteristic_vs_Alpha", "characteristic_vs_alpha.csv", "A1:L12"],
  ["Existing_EFI", "existing_efi.csv", "A1:J18"],
  ["Master_Table", "core10_predictive_alpha_decay_master.csv", "A1:L12"],
  ["EFI_Correlations", "efi_correlations.csv", "A1:I18"],
  ["LOO_Sensitivity", "loo_sensitivity.csv", "A1:L18"],
  ["Feature_Diagnostics", "feature_diagnostics.csv", "A1:L12"],
];

const workbook = Workbook.create();
for (const [sheetName, csvName] of sheets) {
  const csvText = await fs.readFile(path.join(outputRoot, csvName), "utf8");
  await workbook.fromCSV(csvText, { sheetName });
}

for (const [sheetName] of sheets) {
  const sheet = workbook.worksheets.getItem(sheetName);
  const used = sheet.getUsedRange(true);
  const header = used.getRow(0);
  const headerValues = header.values[0] ?? [];
  const typedValues = used.values;
  for (let row = 1; row < typedValues.length; row += 1) {
    for (let column = 0; column < typedValues[row].length; column += 1) {
      typedValues[row][column] = typedCellValue(
        typedValues[row][column],
        String(headerValues[column] ?? "").toLowerCase(),
      );
    }
  }
  used.values = typedValues;
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  if (sheetName.startsWith("Monthly_") || sheetName === "Master_Table") {
    sheet.freezePanes.freezeColumns(2);
  }
  used.format.font = { name: "Aptos", size: 10, color: "#17324D" };
  used.format.rowHeight = 19;
  header.format = {
    fill: "#17324D",
    font: { name: "Aptos", size: 10, bold: true, color: "#FFFFFF" },
    wrapText: true,
    rowHeight: 62,
    borders: { bottom: { style: "medium", color: "#17324D" } },
  };
  for (let index = 0; index < headerValues.length; index += 1) {
    const name = String(headerValues[index] ?? "").toLowerCase();
    const column = used.getColumn(index);
    column.format.columnWidth = columnWidth(name);
    if (isDateColumn(name)) {
      column.format.numberFormat = "yyyy-mm-dd";
    } else if (isIntegerColumn(name)) {
      column.format.numberFormat = "#,##0";
    } else if (isNumericColumn(name)) {
      column.format.numberFormat = "0.000000";
    }
    if (isLongTextColumn(name)) {
      column.format.wrapText = true;
    }
  }
  if (sheetName === "README") {
    sheet.getRange("A2:A18").format = {
      fill: "#E8EEF3",
      font: { bold: true, color: "#17324D" },
    };
    sheet.getRange("B2:B18").format.wrapText = true;
    sheet.getRange("A1:B18").format.rowHeight = 34;
    sheet.getRange("A1:B1").format.rowHeight = 38;
  }
  if (sheetName === "Timing_Audit") {
    used.format.rowHeight = 42;
    used.format.wrapText = true;
  }
  if (sheetName === "Existing_EFI") {
    used.format.rowHeight = 44;
  }
  if (sheetName === "Feature_Diagnostics") {
    used.format.rowHeight = 36;
  }
  if (sheetName === "Master_Table" || sheetName === "Alpha_Persistence_Metrics") {
    header.format.rowHeight = 82;
  }
}

const correlationSheet = workbook.worksheets.getItem("EFI_Correlations");
const correlationHeader = correlationSheet.getUsedRange(true).getRow(0).values[0];
const correlationIndex = correlationHeader.findIndex(
  (value) => String(value) === "correlation",
);
if (correlationIndex >= 0) {
  correlationSheet
    .getUsedRange(true)
    .getColumn(correlationIndex)
    .conditionalFormats.add("colorScale", {
      colors: ["#F4CCCC", "#FFF2CC", "#D9EAD3"],
      thresholds: ["min", { type: "num", value: 0 }, "max"],
    });
}

const alphaSheet = workbook.worksheets.getItem("Alpha_Persistence_Metrics");
const alphaHeader = alphaSheet.getUsedRange(true).getRow(0).values[0];
for (const flagName of [
  "fmb_nonmonotonic_recovery_flag",
  "rankic_nonmonotonic_recovery_flag",
]) {
  const index = alphaHeader.findIndex((value) => String(value) === flagName);
  if (index >= 0) {
    alphaSheet
      .getUsedRange(true)
      .getColumn(index)
      .conditionalFormats.add("containsText", {
        text: "TRUE",
        format: { fill: "#FFF2CC", font: { bold: true, color: "#7F6000" } },
      });
  }
}

await fs.mkdir(outputRoot, { recursive: true });
await fs.mkdir(previewRoot, { recursive: true });

const keyInspection = await workbook.inspect({
  kind: "table",
  sheetId: "Master_Table",
  range: "A1:L12",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 12,
  maxChars: 8000,
});
console.log("KEY_INSPECTION");
console.log(keyInspection.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log("FORMULA_ERROR_SCAN");
console.log(errors.ndjson);

for (const [sheetName, , previewRange] of sheets) {
  const preview = await workbook.render({
    sheetName,
    range: previewRange,
    scale: 1,
    format: "png",
  });
  const previewBytes = new Uint8Array(await preview.arrayBuffer());
  await fs.writeFile(path.join(previewRoot, `${sheetName}.png`), previewBytes);
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
await fs.rm(`${outputPath}.inspect.ndjson`, { force: true });
console.log(`WORKBOOK=${outputPath}`);
console.log(`PREVIEWS=${previewRoot}`);
console.log(`SHEETS=${sheets.length}`);


function isDateColumn(name) {
  return name.includes("eom") || name.endsWith("_date") || name.includes("date_");
}


function isIntegerColumn(name) {
  return (
    name === "horizon" ||
    name === "year" ||
    name.startsWith("n_") ||
    name.endsWith("_count") ||
    name.endsWith("_months") ||
    name.endsWith("_stock_months") ||
    name.endsWith("_observations") ||
    name.includes("cross_section_n") ||
    name.includes("importance_rank") ||
    name.includes("rank_high_to_low")
  );
}


function isNumericColumn(name) {
  return (
    name.includes("correlation") ||
    name.includes("persistence") ||
    name.includes("rank_ic") ||
    name.includes("pearson_ic") ||
    name.includes("fmb") ||
    name.includes("slope") ||
    name.includes("intercept") ||
    name.includes("standard_error") ||
    name.includes("_se") ||
    name.includes("t_stat") ||
    name.includes("fraction") ||
    name.includes("rate") ||
    name.includes("mean") ||
    name.includes("median") ||
    name.includes("std") ||
    name.includes("utility") ||
    name.includes("apr_") ||
    name.includes("sdm") ||
    name.includes("sfm") ||
    name.includes("ssh") ||
    name.includes("absolute_change")
  );
}


function isLongTextColumn(name) {
  return (
    name.includes("path") ||
    name.includes("source") ||
    name.includes("metric") ||
    name.includes("definition") ||
    name.includes("note") ||
    name.includes("interpretation") ||
    name.includes("policy") ||
    name.includes("rule") ||
    name.includes("purpose") ||
    name.includes("detail") ||
    name.includes("timing") ||
    name.includes("classification") ||
    name.includes("join_keys") ||
    name.includes("target_type")
  );
}


function columnWidth(name) {
  if (name === "feature" || name === "theme") return 24;
  if (name.includes("path") || name.includes("source")) return 58;
  if (name.includes("definition") || name.includes("metric")) return 54;
  if (name.includes("note") || name.includes("detail") || name.includes("purpose")) return 64;
  if (name.includes("rule") || name.includes("policy") || name.includes("interpretation")) return 58;
  if (name.includes("method")) return 24;
  if (isDateColumn(name)) return 16;
  if (isIntegerColumn(name)) return 15;
  if (isNumericColumn(name)) return 18;
  return 22;
}


function typedCellValue(value, headerName) {
  if (typeof value !== "string") return value;
  const trimmed = value.trim();
  if (trimmed === "") return null;
  if (isDateColumn(headerName) && /^\d{4}-\d{2}-\d{2}$/.test(trimmed)) {
    return new Date(`${trimmed}T00:00:00Z`);
  }
  if (trimmed.toLowerCase() === "true") return true;
  if (trimmed.toLowerCase() === "false") return false;
  if (/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(trimmed)) {
    const numeric = Number(trimmed);
    if (Number.isFinite(numeric)) return numeric;
  }
  return value;
}
