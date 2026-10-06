import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const projectRoot = process.cwd();
const dataPath = path.join(
  projectRoot,
  ".artifact_tool_final_results",
  "workbook_data.json",
);
const outputDir = path.join(projectRoot, "reports", "final_results");
const previewDir = path.join(
  projectRoot,
  ".artifact_tool_final_results",
  "previews",
);
const outputPath = path.join(outputDir, "final_performance_tables.xlsx");
const payload = JSON.parse(await fs.readFile(dataPath, "utf8"));

const workbook = Workbook.create();
const sheets = {
  lock: workbook.worksheets.add("Specification Lock"),
  main: workbook.worksheets.add("Main Metrics"),
  inference: workbook.worksheets.add("Main Utility Inference"),
  robustness: workbook.worksheets.add("Robustness Metrics"),
  deltas: workbook.worksheets.add("Pairwise Deltas"),
  risk: workbook.worksheets.add("Risk Calibration"),
  sources: workbook.worksheets.add("Sources"),
};

const theme = {
  ink: "#1F2937",
  teal: "#0F766E",
  tealLight: "#D9F0EC",
  blueLight: "#E5EDF8",
  gold: "#B7791F",
  goldLight: "#FCF1D5",
  redLight: "#FCE8E6",
  greenLight: "#E3F3E8",
  gray: "#6B7280",
  grayLight: "#F3F4F6",
  border: "#D1D5DB",
  white: "#FFFFFF",
};

function excelColumn(index) {
  let value = index + 1;
  let result = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    value = Math.floor((value - 1) / 26);
  }
  return result;
}

function writeTitle(sheet, title, subtitle, columnCount) {
  const lastColumn = excelColumn(columnCount - 1);
  sheet.mergeCells(`A1:${lastColumn}1`);
  sheet.getRange("A1").values = [[title]];
  sheet.getRange(`A1:${lastColumn}1`).format = {
    fill: theme.ink,
    font: { bold: true, color: theme.white, size: 16 },
    verticalAlignment: "center",
  };
  sheet.getRange(`A1:${lastColumn}1`).format.rowHeight = 30;
  sheet.mergeCells(`A2:${lastColumn}2`);
  sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange(`A2:${lastColumn}2`).format = {
    fill: theme.grayLight,
    font: { color: theme.gray, italic: true, size: 10 },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.getRange(`A2:${lastColumn}2`).format.rowHeight = 30;
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(4);
}

function writeTable(sheet, headers, rows, options = {}) {
  const headerRow = options.headerRow ?? 4;
  const startRow = headerRow + 1;
  const lastColumn = excelColumn(headers.length - 1);
  sheet.getRange(`A${headerRow}:${lastColumn}${headerRow}`).values = [headers];
  sheet.getRange(`A${headerRow}:${lastColumn}${headerRow}`).format = {
    fill: options.headerFill ?? theme.teal,
    font: { bold: true, color: theme.white, size: 10 },
    wrapText: true,
    verticalAlignment: "center",
    borders: { preset: "outside", style: "thin", color: theme.border },
  };
  sheet.getRange(`A${headerRow}:${lastColumn}${headerRow}`).format.rowHeight = 31;
  if (rows.length > 0) {
    const endRow = startRow + rows.length - 1;
    sheet.getRange(`A${startRow}:${lastColumn}${endRow}`).values = rows;
    sheet.getRange(`A${startRow}:${lastColumn}${endRow}`).format = {
      verticalAlignment: "center",
      borders: {
        insideHorizontal: { style: "thin", color: theme.border },
        bottom: { style: "thin", color: theme.border },
      },
    };
    if (options.bodyRowHeight) {
      sheet.getRange(`A${startRow}:${lastColumn}${endRow}`).format.rowHeight =
        options.bodyRowHeight;
    }
  }
  return { headerRow, startRow, endRow: startRow + rows.length - 1, lastColumn };
}

function applyColumnWidths(sheet, widths) {
  widths.forEach((width, index) => {
    const column = excelColumn(index);
    sheet.getRange(`${column}:${column}`).format.columnWidth = width;
  });
}

function applySpecBanding(sheet, specColumn, startRow, endRow) {
  for (let row = startRow; row <= endRow; row += 1) {
    const specification = String(sheet.getRange(`${specColumn}${row}`).values[0][0]);
    let fill = theme.grayLight;
    if (specification === "Core10") fill = theme.tealLight;
    if (specification === "Final15") fill = theme.blueLight;
    if (specification === "Final20 Conservative") fill = theme.goldLight;
    sheet.getRange(`${specColumn}${row}`).format = {
      fill,
      font: { bold: true, color: theme.ink },
    };
  }
}

// Specification lock.
writeTitle(
  sheets.lock,
  "Final Specification Lock-In",
  "Top500/Core10 is the main specification; Final15 and Final20 Conservative are robustness only.",
  4,
);
const lockRows = payload.specification_lock.map((row) => [
  row.Role,
  row.Specification,
  row.Decision,
  row.Use,
]);
const lockTable = writeTable(
  sheets.lock,
  ["Role", "Specification", "Decision", "Permitted use"],
  lockRows,
  { bodyRowHeight: 38 },
);
sheets.lock.getRange(`A${lockTable.startRow}:D${lockTable.endRow}`).format.wrapText = true;
sheets.lock.getRange(`A${lockTable.startRow}:A${lockTable.endRow}`).format.font = {
  bold: true,
};
sheets.lock.getRange(`A${lockTable.startRow}:D${lockTable.startRow}`).format.fill = theme.tealLight;
applyColumnWidths(sheets.lock, [16, 35, 28, 75]);

// Main metrics.
writeTitle(
  sheets.main,
  "Main Top500/Core10 Performance",
  "Annualized return, volatility, trading cost and utility; turnover is average monthly turnover.",
  8,
);
const mainRows = payload.main_metrics.map((row) => [
  row.method,
  row.annualized_net_return,
  row.annualized_volatility,
  row.net_sharpe,
  row.average_turnover,
  row.average_leverage,
  row.annualized_trading_cost,
  row.annualized_ex_ante_utility_flow,
]);
const mainTable = writeTable(
  sheets.main,
  ["Method", "Net return", "Volatility", "Sharpe", "Turnover", "Leverage", "Trading cost", "Utility"],
  mainRows,
  { bodyRowHeight: 24 },
);
sheets.main.getRange(`B${mainTable.startRow}:C${mainTable.endRow}`).format.numberFormat = "0.00%";
sheets.main.getRange(`D${mainTable.startRow}:D${mainTable.endRow}`).format.numberFormat = "0.000";
sheets.main.getRange(`E${mainTable.startRow}:E${mainTable.endRow}`).format.numberFormat = "0.00%";
sheets.main.getRange(`F${mainTable.startRow}:F${mainTable.endRow}`).format.numberFormat = "0.000";
sheets.main.getRange(`G${mainTable.startRow}:H${mainTable.endRow}`).format.numberFormat = "0.00%";
sheets.main.getRange(`A${mainTable.startRow}:H${mainTable.endRow}`).format.fill = theme.tealLight;
applyColumnWidths(sheets.main, [30, 15, 15, 12, 14, 13, 16, 14]);

// Main utility inference.
writeTitle(
  sheets.inference,
  "Main Utility Inference",
  "Top500/Core10 monthly utility-flow differences; HAC and block-bootstrap outputs are read from the completed experiment.",
  9,
);
const inferenceRows = payload.main_inference.map((row) => [
  row.comparison,
  row.annualized_mean_difference,
  row.newey_west_t_stat,
  row.bootstrap_annualized_ci_lower,
  row.bootstrap_annualized_ci_upper,
  row.probability_delta_positive,
  row.annual_years_won,
  row.annual_years_total,
  row.years_won,
]);
const inferenceTable = writeTable(
  sheets.inference,
  ["Comparison", "Annual utility difference", "HAC t-stat", "Bootstrap CI lower", "Bootstrap CI upper", "P(delta > 0)", "Years won", "Years total", "Years won label"],
  inferenceRows,
  { bodyRowHeight: 26 },
);
sheets.inference.getRange(`B${inferenceTable.startRow}:B${inferenceTable.endRow}`).format.numberFormat = "0.00%";
sheets.inference.getRange(`C${inferenceTable.startRow}:C${inferenceTable.endRow}`).format.numberFormat = "0.000";
sheets.inference.getRange(`D${inferenceTable.startRow}:F${inferenceTable.endRow}`).format.numberFormat = "0.00%";
sheets.inference.getRange(`G${inferenceTable.startRow}:H${inferenceTable.endRow}`).format.numberFormat = "0";
sheets.inference.getRange(`A${inferenceTable.startRow}:I${inferenceTable.endRow}`).format.fill = theme.tealLight;
sheets.inference.getRange(`C${inferenceTable.startRow}:C${inferenceTable.endRow}`).conditionalFormats.add(
  "cellIs",
  { operator: "greaterThanOrEqual", formula: 1.96, format: { fill: theme.greenLight, font: { bold: true, color: theme.teal } } },
);
applyColumnWidths(sheets.inference, [40, 21, 13, 18, 18, 16, 12, 12, 17]);

// Robustness metrics.
writeTitle(
  sheets.robustness,
  "Performance Across Locked Specifications",
  "Core10 is main; Final15 and Final20 Conservative are robustness specifications.",
  9,
);
const robustnessRows = payload.robustness_metrics.map((row) => [
  row.specification,
  row.method,
  row.annualized_net_return,
  row.annualized_volatility,
  row.net_sharpe,
  row.average_turnover,
  row.average_leverage,
  row.annualized_trading_cost,
  row.annualized_ex_ante_utility_flow,
]);
const robustnessTable = writeTable(
  sheets.robustness,
  ["Specification", "Method", "Net return", "Volatility", "Sharpe", "Turnover", "Leverage", "Trading cost", "Utility"],
  robustnessRows,
  { bodyRowHeight: 23 },
);
sheets.robustness.getRange(`C${robustnessTable.startRow}:D${robustnessTable.endRow}`).format.numberFormat = "0.00%";
sheets.robustness.getRange(`E${robustnessTable.startRow}:E${robustnessTable.endRow}`).format.numberFormat = "0.000";
sheets.robustness.getRange(`F${robustnessTable.startRow}:F${robustnessTable.endRow}`).format.numberFormat = "0.00%";
sheets.robustness.getRange(`G${robustnessTable.startRow}:G${robustnessTable.endRow}`).format.numberFormat = "0.000";
sheets.robustness.getRange(`H${robustnessTable.startRow}:I${robustnessTable.endRow}`).format.numberFormat = "0.00%";
applySpecBanding(sheets.robustness, "A", robustnessTable.startRow, robustnessTable.endRow);
applyColumnWidths(sheets.robustness, [24, 30, 14, 14, 12, 14, 13, 16, 14]);

// Pairwise deltas with formula references to the robustness sheet.
writeTitle(
  sheets.deltas,
  "Pairwise Robustness Differences",
  "Each delta equals the right-hand specification minus the left-hand specification and is formula-linked to Robustness Metrics.",
  11,
);
const robustnessRowMap = new Map();
payload.robustness_metrics.forEach((row, index) => {
  robustnessRowMap.set(`${row.specification}|${row.method}`, robustnessTable.startRow + index);
});
const deltaBaseRows = payload.pairwise_deltas.map((row) => [
  row.comparison,
  row.left_specification,
  row.right_specification,
  row.method,
  null,
  null,
  null,
  null,
  null,
  null,
  null,
]);
const deltaTable = writeTable(
  sheets.deltas,
  ["Comparison", "Left specification", "Right specification", "Method", "Delta net return", "Delta volatility", "Delta Sharpe", "Delta turnover", "Delta leverage", "Delta trading cost", "Delta utility"],
  deltaBaseRows,
  { bodyRowHeight: 24 },
);
payload.pairwise_deltas.forEach((row, index) => {
  const outputRow = deltaTable.startRow + index;
  const leftRow = robustnessRowMap.get(`${row.left_specification}|${row.method}`);
  const rightRow = robustnessRowMap.get(`${row.right_specification}|${row.method}`);
  const sourceColumns = ["C", "D", "E", "F", "G", "H", "I"];
  const formulas = sourceColumns.map(
    (column) => `='Robustness Metrics'!${column}${rightRow}-'Robustness Metrics'!${column}${leftRow}`,
  );
  sheets.deltas.getRange(`E${outputRow}:K${outputRow}`).formulas = [formulas];
});
sheets.deltas.getRange(`E${deltaTable.startRow}:F${deltaTable.endRow}`).format.numberFormat = "0.00%";
sheets.deltas.getRange(`G${deltaTable.startRow}:G${deltaTable.endRow}`).format.numberFormat = "0.000";
sheets.deltas.getRange(`H${deltaTable.startRow}:H${deltaTable.endRow}`).format.numberFormat = "0.00%";
sheets.deltas.getRange(`I${deltaTable.startRow}:I${deltaTable.endRow}`).format.numberFormat = "0.000";
sheets.deltas.getRange(`J${deltaTable.startRow}:K${deltaTable.endRow}`).format.numberFormat = "0.00%";
sheets.deltas.getRange(`K${deltaTable.startRow}:K${deltaTable.endRow}`).conditionalFormats.add(
  "cellIs",
  { operator: "greaterThan", formula: 0, format: { fill: theme.greenLight, font: { color: theme.teal } } },
);
sheets.deltas.getRange(`K${deltaTable.startRow}:K${deltaTable.endRow}`).conditionalFormats.add(
  "cellIs",
  { operator: "lessThan", formula: 0, format: { fill: theme.redLight, font: { color: "#B42318" } } },
);
applyColumnWidths(sheets.deltas, [34, 22, 24, 29, 17, 17, 15, 16, 16, 19, 15]);

// Risk calibration.
writeTitle(
  sheets.risk,
  "Risk Calibration Across Specifications",
  "Calibration band is realized/predicted annualized volatility in [0.6, 1.5]. Static exceptions are disclosed rather than retuned.",
  7,
);
const riskRows = payload.risk_calibration.map((row) => [
  row.specification,
  row.method,
  row.yearly_ratio_median,
  row.yearly_ratio_max,
  row.fraction_years_within_0_6_1_5,
  row.years_ratio_above_2,
  row.quality_gate_passed ? "PASS" : "FAIL",
]);
const riskTable = writeTable(
  sheets.risk,
  ["Specification", "Method", "Median ratio", "Max ratio", "Years within [0.6, 1.5]", "Years ratio > 2", "Method gate"],
  riskRows,
  { bodyRowHeight: 24 },
);
sheets.risk.getRange(`C${riskTable.startRow}:D${riskTable.endRow}`).format.numberFormat = "0.000";
sheets.risk.getRange(`E${riskTable.startRow}:E${riskTable.endRow}`).format.numberFormat = "0.0%";
sheets.risk.getRange(`F${riskTable.startRow}:F${riskTable.endRow}`).format.numberFormat = "0";
applySpecBanding(sheets.risk, "A", riskTable.startRow, riskTable.endRow);
sheets.risk.getRange(`G${riskTable.startRow}:G${riskTable.endRow}`).conditionalFormats.add(
  "containsText",
  { text: "PASS", format: { fill: theme.greenLight, font: { bold: true, color: theme.teal } } },
);
sheets.risk.getRange(`G${riskTable.startRow}:G${riskTable.endRow}`).conditionalFormats.add(
  "containsText",
  { text: "FAIL", format: { fill: theme.redLight, font: { bold: true, color: "#B42318" } } },
);
applyColumnWidths(sheets.risk, [24, 30, 15, 14, 24, 18, 14]);

// Sources and definitions.
writeTitle(
  sheets.sources,
  "Sources and Definitions",
  "All values are read from completed experiment outputs. No model training or backtest is performed by this consolidation workbook.",
  4,
);
const sourceRows = payload.sources.map((row) => [
  row.Specification,
  row["Metrics source"],
  row["Inference source"],
  row["Risk source"],
]);
const sourceTable = writeTable(
  sheets.sources,
  ["Specification", "Metrics source", "Inference source", "Risk source"],
  sourceRows,
  { bodyRowHeight: 42 },
);
sheets.sources.getRange(`A${sourceTable.startRow}:D${sourceTable.endRow}`).format.wrapText = true;
const definitionStart = sourceTable.endRow + 3;
sheets.sources.getRange(`A${definitionStart}:D${definitionStart}`).merge();
sheets.sources.getRange(`A${definitionStart}`).values = [["Definitions"]];
sheets.sources.getRange(`A${definitionStart}:D${definitionStart}`).format = {
  fill: theme.gold,
  font: { bold: true, color: theme.white },
};
const definitionRows = Object.entries(payload.definitions).map(([term, definition]) => [
  term,
  definition,
]);
const definitionEnd = definitionStart + definitionRows.length;
sheets.sources.getRange(`A${definitionStart + 1}:B${definitionEnd}`).values = definitionRows;
sheets.sources.getRange(`A${definitionStart + 1}:B${definitionEnd}`).format = {
  wrapText: true,
  borders: { insideHorizontal: { style: "thin", color: theme.border } },
};
sheets.sources.getRange(`A${definitionStart + 1}:A${definitionEnd}`).format.font = { bold: true };
applyColumnWidths(sheets.sources, [32, 75, 75, 75]);

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const keyInspection = await workbook.inspect({
  kind: "table",
  sheetId: "Main Metrics",
  range: `A1:H${mainTable.endRow}`,
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 10,
  maxChars: 5000,
});
console.log("KEY_INSPECTION");
console.log(keyInspection.ndjson);

const formulaInspection = await workbook.inspect({
  kind: "formula",
  sheetId: "Pairwise Deltas",
  range: `E${deltaTable.startRow}:K${deltaTable.endRow}`,
  maxChars: 5000,
  options: { maxResults: 120 },
});
console.log("FORMULA_INSPECTION");
console.log(formulaInspection.ndjson);

const errorInspection = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log("ERROR_SCAN");
console.log(errorInspection.ndjson);

for (const sheetName of Object.keys(sheets).map((key) => sheets[key].name)) {
  const preview = await workbook.render({
    sheetName,
    autoCrop: "all",
    scale: 1,
    format: "png",
  });
  const safeName = sheetName.toLowerCase().replaceAll(" ", "_");
  await fs.writeFile(
    path.join(previewDir, `${safeName}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

const exportedBlob = await FileBlob.load(outputPath);
const reopened = await SpreadsheetFile.importXlsx(exportedBlob);
const reopenedCheck = await reopened.inspect({
  kind: "table",
  sheetId: "Main Utility Inference",
  range: `A4:I${inferenceTable.endRow}`,
  include: "values,formulas",
  tableMaxRows: 10,
  tableMaxCols: 10,
  maxChars: 5000,
});
console.log("REOPENED_CHECK");
console.log(reopenedCheck.ndjson);
const reopenedErrors = await reopened.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "reopened workbook formula error scan",
});
console.log("REOPENED_ERROR_SCAN");
console.log(reopenedErrors.ndjson);
console.log(`output=${outputPath}`);
console.log(`preview_dir=${previewDir}`);
