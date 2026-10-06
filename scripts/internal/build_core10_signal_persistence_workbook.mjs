import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const outputRoot = path.join(projectRoot, "outputs", "extensions", "core10_signal_persistence");
const outputPath = path.join(outputRoot, "core10_signal_persistence_analysis.xlsx");
const previewRoot = path.join(os.tmpdir(), "core10_signal_persistence_workbook_previews");

const sheets = [
  ["README", "workbook_readme.csv", "A1:B15"],
  ["Input_Audit", "input_audit.csv", "A1:E11"],
  ["Feature_Persistence", "feature_persistence.csv", "A1:AF12"],
  ["Monthly_Persistence", "monthly_persistence.csv", "A1:G18"],
  ["Feature_IC", "feature_ic.csv", "A1:N12"],
  ["Existing_EFI", "existing_efi.csv", "A1:J18"],
  ["Master_Table", "core10_feature_persistence_master.csv", "A1:AZ12"],
  ["Correlation_Analysis", "correlation_analysis.csv", "A1:G18"],
  ["Controlled_Regression", "controlled_regression.csv", "A1:K9"],
  ["LOO_Sensitivity", "loo_sensitivity.csv", "A1:G18"],
  ["High_Low_Persistence", "high_low_persistence.csv", "A1:J6"],
  ["Theme_Analysis", "theme_analysis.csv", "A1:J10"],
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
  const typedValues = used.values;
  for (let row = 1; row < typedValues.length; row += 1) {
    for (let column = 0; column < typedValues[row].length; column += 1) {
      typedValues[row][column] = typedCellValue(typedValues[row][column]);
    }
  }
  used.values = typedValues;
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  used.format.font = { name: "Aptos", size: 10, color: "#17324D" };
  used.format.rowHeight = 19;
  header.format = {
    fill: "#17324D",
    font: { name: "Aptos", size: 10, bold: true, color: "#FFFFFF" },
    wrapText: true,
    rowHeight: 58,
    borders: { bottom: { style: "medium", color: "#17324D" } },
  };
  const headerValues = header.values[0] ?? [];
  for (let index = 0; index < headerValues.length; index += 1) {
    const name = String(headerValues[index] ?? "").toLowerCase();
    const column = used.getColumn(index);
    column.format.columnWidth = columnWidth(name);
    if (isIntegerColumn(name)) {
      column.format.numberFormat = "#,##0";
    } else if (isNumericColumn(name)) {
      column.format.numberFormat = "0.000000";
    }
    if (isLongTextColumn(name)) {
      column.format.wrapText = true;
    }
  }
  if (sheetName === "README") {
    sheet.getRange("A2:A15").format = {
      fill: "#E8EEF3",
      font: { bold: true, color: "#17324D" },
    };
    sheet.getRange("B2:B15").format.wrapText = true;
    sheet.getRange("A1:B15").format.rowHeight = 30;
    sheet.getRange("A1:B1").format.rowHeight = 34;
  }
  if (sheetName === "Input_Audit") {
    sheet.getRange("A2:E10").format.wrapText = true;
    sheet.getRange("A2:E10").format.rowHeight = 62;
  }
  if (sheetName === "Existing_EFI") {
    sheet.getRange("A2:J21").format.rowHeight = 36;
  }
  if (sheetName === "Master_Table") {
    sheet.getRange("A2:AZ11").format.rowHeight = 44;
  }
  if (sheetName === "Feature_IC") {
    sheet.getRange("A2:M11").format.rowHeight = 34;
  }
  if (sheetName === "Correlation_Analysis") {
    sheet.getRange("A2:G17").format.rowHeight = 34;
  }
  if (sheetName === "Controlled_Regression") {
    sheet.getRange("A2:K7").format.rowHeight = 34;
  }
  if (sheetName === "Theme_Analysis") {
    sheet.getRange("A2:J8").format.wrapText = true;
    sheet.getRange("A2:J8").format.rowHeight = 44;
  }
  if (sheetName === "Master_Table" || sheetName === "Feature_Persistence") {
    header.format.rowHeight = 72;
  }
}

const consistencySheet = workbook.worksheets.getItem("Feature_Persistence");
const consistencyHeader = consistencySheet.getUsedRange(true).getRow(0).values[0];
const consistencyIndex = consistencyHeader.findIndex(
  (value) => String(value) === "persistence_consistency_flag",
);
if (consistencyIndex >= 0) {
  const consistencyColumn = consistencySheet.getUsedRange(true).getColumn(consistencyIndex);
  consistencyColumn.conditionalFormats.add("containsText", {
    text: "review",
    format: { fill: "#FDE8E7", font: { bold: true, color: "#A33A32" } },
  });
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
  maxChars: 7000,
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
console.log(`WORKBOOK=${outputPath}`);
console.log(`PREVIEWS=${previewRoot}`);
console.log(`SHEETS=${sheets.length}`);


function isIntegerColumn(name) {
  return (
    name.startsWith("n_") ||
    name.endsWith("_count") ||
    name.endsWith("_months") ||
    name.endsWith("_pairs") ||
    name.endsWith("_transitions") ||
    name === "year"
  );
}


function isNumericColumn(name) {
  return (
    name.includes("correlation") ||
    name.includes("persistence") ||
    name.includes("rho") ||
    name.includes("phi") ||
    name.includes("half_life") ||
    name.includes("standard_error") ||
    name.includes("t_stat") ||
    name.includes("r_squared") ||
    name.includes("fraction") ||
    name.includes("rate") ||
    name.includes("mean") ||
    name.includes("median") ||
    name.includes("std") ||
    name.includes("p10") ||
    name.includes("p90") ||
    name.includes("ic") ||
    name.includes("utility") ||
    name.includes("coefficient") ||
    name.includes("error") ||
    name.includes("condition_number")
  );
}


function isLongTextColumn(name) {
  return (
    name.includes("path") ||
    name.includes("source") ||
    name.includes("metric") ||
    name.includes("definition") ||
    name.includes("note") ||
    name.includes("features") ||
    name.includes("interpretation") ||
    name.includes("decision") ||
    name.includes("use") ||
    name.includes("detail") ||
    name.includes("timing") ||
    name === "p_value_method" ||
    name === "dependent_variable"
  );
}


function columnWidth(name) {
  if (name === "feature" || name === "item_name" || name === "theme") return 24;
  if (name.includes("path") || name.includes("source")) return 62;
  if (name.includes("definition") || name.includes("metric")) return 58;
  if (name.includes("note") || name.includes("features") || name.includes("detail")) return 70;
  if (name.includes("method") || name.includes("specification")) return 25;
  if (name.includes("relationship")) return 34;
  if (name.includes("timing") || name === "p_value_method" || name === "dependent_variable") return 58;
  if (name.includes("date") || name.includes("eom")) return 15;
  if (isIntegerColumn(name)) return 14;
  if (isNumericColumn(name)) return 18;
  return 22;
}


function typedCellValue(value) {
  if (typeof value !== "string") return value;
  const trimmed = value.trim();
  if (trimmed === "") return null;
  if (trimmed.toLowerCase() === "true") return true;
  if (trimmed.toLowerCase() === "false") return false;
  if (/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(trimmed)) {
    const numeric = Number(trimmed);
    if (Number.isFinite(numeric)) return numeric;
  }
  return value;
}
