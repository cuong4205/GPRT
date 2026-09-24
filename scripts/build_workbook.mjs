import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const inputPath = process.argv[2] ?? "outputs/vrptw_baselines/results.json";
const outputPath = process.argv[3] ?? "outputs/vrptw_baselines/vrptw_solomon_baselines.xlsx";
const payload = JSON.parse(await fs.readFile(inputPath, "utf8"));
const workbook = Workbook.create();
const fontName = "Arial";
const colors = {
  navy: "#17365D",
  blue: "#2F75B5",
  lightBlue: "#D9EAF7",
  pale: "#F4F7FA",
  green: "#E2F0D9",
  amber: "#FFF2CC",
  red: "#FCE4D6",
  gray: "#666666",
  white: "#FFFFFF",
};

function colLetter(index) {
  let result = "";
  let value = index + 1;
  while (value > 0) {
    value -= 1;
    result = String.fromCharCode(65 + (value % 26)) + result;
    value = Math.floor(value / 26);
  }
  return result;
}

function applyBase(sheet, usedRange) {
  sheet.showGridLines = false;
  usedRange.format.font = { name: fontName, size: 10, color: "#222222" };
  usedRange.format.verticalAlignment = "center";
}

function addDataSheet({ name, title, records, columns, tableName, widths = {}, formats = {} }) {
  const sheet = workbook.worksheets.add(name);
  sheet.getRange("A2").values = [[title]];
  sheet.getRange("A2").format.font = { name: fontName, size: 14, bold: true, color: colors.navy };
  sheet.getRange("A3").values = [[`${records.length.toLocaleString("vi-VN")} bản ghi`]];
  sheet.getRange("A3").format.font = { name: fontName, size: 10, italic: true, color: colors.gray };
  const headers = columns.map((column) => column.label);
  const values = records.map((record) => columns.map((column) => {
    const value = column.value ? column.value(record) : record[column.key];
    return value === undefined ? null : value;
  }));
  const lastColumn = colLetter(columns.length - 1);
  sheet.getRange(`A5:${lastColumn}5`).values = [headers];
  if (values.length) {
    sheet.getRangeByIndexes(5, 0, values.length, columns.length).values = values;
    const table = sheet.tables.add(`A5:${lastColumn}${5 + values.length}`, true, tableName);
    table.style = "TableStyleMedium2";
    table.showBandedRows = true;
  }
  const used = sheet.getRange(`A2:${lastColumn}${Math.max(5, 5 + values.length)}`);
  applyBase(sheet, used);
  sheet.getRange(`A5:${lastColumn}5`).format = {
    fill: colors.navy,
    font: { name: fontName, size: 10, bold: true, color: colors.white },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
  sheet.freezePanes.freezeRows(5);
  for (let i = 0; i < columns.length; i += 1) {
    const letter = colLetter(i);
    sheet.getRange(`${letter}:${letter}`).format.columnWidth = widths[columns[i].key] ?? 14;
    if (formats[columns[i].key]) {
      sheet.getRange(`${letter}6:${letter}${5 + values.length}`).format.numberFormat = formats[columns[i].key];
    }
  }
  return sheet;
}

const summary = workbook.worksheets.add("Tổng quan");
summary.tabColor = colors.navy;
summary.showGridLines = false;
summary.getRange("A2").values = [["Baseline VRPTW trên bộ Solomon"]];
summary.getRange("A2").format.font = { name: fontName, size: 16, bold: true, color: colors.navy };
summary.getRange("A3").values = [["Kết quả chính trên test split cố định; 5 seed cho mỗi phương pháp"]];
summary.getRange("A3").format.font = { name: fontName, size: 10, italic: true, color: colors.gray };
summary.getRange("A5:H5").values = [["Số instance", payload.metadata.solomon_instance_count, "Train", payload.metadata.train_count, "Validation", payload.metadata.validation_count, "Test", payload.metadata.test_count]];
summary.getRange("A5:H5").format = { fill: colors.lightBlue, font: { name: fontName, size: 10, bold: true, color: colors.navy } };
summary.getRange("A7:M7").values = [["Hạng", "Phương pháp", "Số lượt", "Khả thi TB", "SD khả thi", "Số xe TB", "SD số xe", "Khoảng cách TB", "SD khoảng cách", "Chờ TB", "SD chờ", "Thời gian áp dụng (ms)", "SD thời gian (ms)"]];
const summaryRows = payload.method_summary.map((row) => [
  row.rank, row.method, row.runs, row.feasible_rate_mean, row.feasible_rate_std,
  row.mean_vehicles_mean, row.mean_vehicles_std, row.mean_distance_mean,
  row.mean_distance_std, row.mean_waiting_time_mean, row.mean_waiting_time_std,
  row.total_runtime_ms_mean, row.total_runtime_ms_std,
]);
summary.getRange("A8:M12").values = summaryRows;
summary.tables.add("A7:M12", true, "MethodSummary").style = "TableStyleMedium2";
summary.getRange("A7:M7").format = { fill: colors.navy, font: { name: fontName, size: 10, bold: true, color: colors.white }, horizontalAlignment: "center", wrapText: true };
summary.getRange("D8:E12").format.numberFormat = "0.00%";
summary.getRange("F8:M12").format.numberFormat = "#,##0.00";
summary.getRange("A14:B18").values = [
  ["Kết luận", "Quan sát từ dữ liệu test"],
  ["EDD", "Đạt 100% khả thi, 11,00 xe và 2.407,35 khoảng cách; baseline thực dụng nhất."],
  ["GP", "Cả 5 seed chọn biểu thức due, tái khám phá EDD nhưng không cải thiện chất lượng."],
  ["Greedy", "Khoảng cách thấp trên phần nghiệm khả thi nhưng chỉ đạt 75% khả thi."],
  ["Random", "Biến động và yếu nhất: 53,33% khả thi, xác nhận cần luật ưu tiên có cấu trúc."],
];
summary.getRange("A14:B14").format = { fill: colors.navy, font: { name: fontName, size: 10, bold: true, color: colors.white } };
summary.getRange("A15:A18").format.font = { name: fontName, size: 10, bold: true, color: colors.navy };
summary.getRange("B15:B18").format.wrapText = true;
summary.getRange("A20:B24").values = [
  ["Quy ước", "Giá trị"],
  ["Mục tiêu", "Khả thi → số xe → tổng khoảng cách Euclid không làm tròn"],
  ["Trung bình xe/khoảng cách", "Chỉ tính trên nghiệm khả thi; tỷ lệ khả thi báo cáo riêng"],
  ["Seed", "20260914–20260918"],
  ["BKS", "Không có trong thư mục dữ liệu; không tính optimality gap"],
];
summary.getRange("A20:B20").format = { fill: colors.navy, font: { name: fontName, size: 10, bold: true, color: colors.white } };
summary.getRange("A2:M24").format.font = { name: fontName, size: 10, color: "#222222" };
summary.getRange("A2").format.font = { name: fontName, size: 16, bold: true, color: colors.navy };
summary.getRange("A:A").format.columnWidth = 18;
summary.getRange("B:B").format.columnWidth = 76;
summary.getRange("C:M").format.columnWidth = 15;
summary.getRange("B15:B24").format.wrapText = true;
summary.getRange("A7:M7").format = { fill: colors.navy, font: { name: fontName, size: 10, bold: true, color: colors.white }, horizontalAlignment: "center", wrapText: true };
summary.getRange("A14:B14").format = { fill: colors.navy, font: { name: fontName, size: 10, bold: true, color: colors.white } };
summary.getRange("A20:B20").format = { fill: colors.navy, font: { name: fontName, size: 10, bold: true, color: colors.white } };

addDataSheet({
  name: "Theo lượt", title: "Kết quả test theo từng lượt chạy", tableName: "RunSummary",
  records: payload.run_summary,
  columns: [
    { key: "method", label: "Phương pháp" }, { key: "run", label: "Lượt" },
    { key: "seed", label: "Seed" }, { key: "test_instances", label: "Số test" },
    { key: "feasible_rate", label: "Tỷ lệ khả thi" }, { key: "mean_vehicles", label: "Số xe TB" },
    { key: "mean_distance", label: "Khoảng cách TB" }, { key: "mean_waiting_time", label: "Chờ TB" },
    { key: "mean_objective", label: "Objective TB" }, { key: "total_runtime_ms", label: "Tổng runtime (ms)" },
  ],
  widths: { method: 15, run: 10, seed: 14, test_instances: 12, feasible_rate: 15, mean_vehicles: 13, mean_distance: 16, mean_waiting_time: 14, mean_objective: 18, total_runtime_ms: 18 },
  formats: { feasible_rate: "0.00%", mean_vehicles: "#,##0.00", mean_distance: "#,##0.00", mean_waiting_time: "#,##0.00", mean_objective: "#,##0.00", total_runtime_ms: "#,##0.00" },
});

addDataSheet({
  name: "Chi tiết", title: "Giá trị tính toán theo instance, phương pháp và seed", tableName: "InstanceDetails",
  records: payload.details,
  columns: [
    { key: "run", label: "Lượt" }, { key: "seed", label: "Seed" }, { key: "method", label: "Phương pháp" },
    { key: "instance", label: "Instance" }, { key: "family", label: "Họ" }, { key: "split", label: "Split" },
    { key: "customers", label: "Khách hàng" }, { key: "max_vehicles", label: "Giới hạn xe" }, { key: "capacity", label: "Tải trọng" },
    { key: "feasible", label: "Khả thi" }, { key: "vehicle_count", label: "Số xe" }, { key: "distance", label: "Khoảng cách" },
    { key: "waiting_time", label: "Thời gian chờ" }, { key: "objective", label: "Objective" }, { key: "runtime_ms", label: "Runtime (ms)" },
    { key: "unserved_count", label: "Chưa phục vụ" }, { key: "unserved", label: "ID chưa phục vụ" }, { key: "expression", label: "Biểu thức GP" },
  ],
  widths: { run: 9, seed: 13, method: 14, instance: 13, family: 9, split: 12, customers: 12, max_vehicles: 12, capacity: 11, feasible: 11, vehicle_count: 10, distance: 15, waiting_time: 15, objective: 18, runtime_ms: 14, unserved_count: 13, unserved: 24, expression: 58 },
  formats: { capacity: "#,##0.00", distance: "#,##0.00", waiting_time: "#,##0.00", objective: "#,##0.00", runtime_ms: "#,##0.000" },
});

addDataSheet({
  name: "Tuyến xe", title: "Chi tiết từng tuyến được dựng", tableName: "RouteDetails",
  records: payload.routes,
  columns: [
    { key: "run", label: "Lượt" }, { key: "seed", label: "Seed" }, { key: "method", label: "Phương pháp" },
    { key: "instance", label: "Instance" }, { key: "route_id", label: "Tuyến" }, { key: "route", label: "Chuỗi khách hàng" },
    { key: "customer_visits", label: "Số khách" }, { key: "load", label: "Tải" }, { key: "distance", label: "Khoảng cách" },
    { key: "end_time", label: "Giờ kết thúc" }, { key: "waiting_time", label: "Thời gian chờ" }, { key: "feasible", label: "Tuyến khả thi" },
  ],
  widths: { run: 9, seed: 13, method: 14, instance: 13, route_id: 9, route: 70, customer_visits: 11, load: 12, distance: 15, end_time: 15, waiting_time: 15, feasible: 13 },
  formats: { load: "#,##0.00", distance: "#,##0.00", end_time: "#,##0.00", waiting_time: "#,##0.00" },
});

addDataSheet({
  name: "GP", title: "Quá trình tìm kiếm Genetic Programming", tableName: "GPRuns",
  records: payload.gp_runs,
  columns: [
    { key: "run", label: "Lượt" }, { key: "seed", label: "Seed" }, { key: "expression", label: "Biểu thức chọn" },
    { key: "tree_size", label: "Số nút" }, { key: "train_fitness", label: "Fitness train" }, { key: "validation_fitness", label: "Fitness validation" },
    { key: "fitness_requests", label: "Fitness requests" }, { key: "unique_expressions", label: "Biểu thức duy nhất" },
    { key: "instance_evaluations", label: "Instance evaluations" }, { key: "cache_hits", label: "Cache hits" },
    { key: "gp_search_seconds", label: "Tìm kiếm (giây)" }, { key: "generation_best", label: "Best theo thế hệ", value: (row) => row.generation_best.map((value) => value.toFixed(3)).join("; ") },
  ],
  widths: { run: 9, seed: 13, expression: 54, tree_size: 10, train_fitness: 18, validation_fitness: 20, fitness_requests: 17, unique_expressions: 18, instance_evaluations: 20, cache_hits: 12, gp_search_seconds: 17, generation_best: 64 },
  formats: { train_fitness: "#,##0.00", validation_fitness: "#,##0.00", gp_search_seconds: "#,##0.00" },
});

addDataSheet({
  name: "Dữ liệu", title: "Manifest bộ Solomon", tableName: "DatasetInventory",
  records: payload.inventory,
  columns: [
    { key: "instance", label: "Instance" }, { key: "source", label: "Nguồn" }, { key: "family", label: "Họ" },
    { key: "customers", label: "Khách hàng" }, { key: "max_vehicles", label: "Giới hạn xe" }, { key: "capacity", label: "Tải trọng" },
    { key: "depot_due_date", label: "Depot due date" }, { key: "split", label: "Split" }, { key: "sha256", label: "SHA-256" },
    { key: "path", label: "Tên file", value: (row) => path.basename(row.path) },
  ],
  widths: { instance: 14, source: 14, family: 10, customers: 12, max_vehicles: 13, capacity: 12, depot_due_date: 15, split: 13, sha256: 68, path: 18 },
  formats: { capacity: "#,##0.00", depot_due_date: "#,##0.00" },
});

const analysis = workbook.worksheets.add("Phân tích");
analysis.tabColor = "#8497B0";
analysis.showGridLines = false;
analysis.getRange("A2").values = [["Phân tích và đánh giá phương pháp"]];
analysis.getRange("A2").format.font = { name: fontName, size: 16, bold: true, color: colors.navy };
const analysisRows = [
  ["Phương pháp", "Đánh giá cụ thể"],
  ["EDD", "Đạt 100% khả thi, 11,00 xe và 2.407,35 khoảng cách. Ưu tiên deadline kiểm soát trực tiếp rủi ro time window; đây là baseline thực dụng nhất."],
  ["GP", "Cả 5 seed đều chọn terminal due nên kết quả trùng EDD. GP nhận ra feature quan trọng nhưng chưa tìm được luật tổ hợp tốt hơn; tìm kiếm offline trung bình 9,72 giây/lượt và áp dụng chậm hơn EDD khoảng 2,01 lần."],
  ["FIFO", "Đạt 83,33% khả thi; thất bại ở R101 và RC101 trong cả 5 lượt. ID khách hàng không biểu diễn vị trí hay độ khẩn cấp, dẫn đến nhiều xe và dễ bế tắc."],
  ["Greedy", "Đạt 75,00% khả thi; thất bại ở R101, RC101, RC102. Quãng đường thấp trên tập nghiệm khả thi riêng nhưng không thể xem là thắng vì bỏ qua các instance bất khả thi và cần nhiều xe hơn EDD."],
  ["Random", "Đạt 53,33% ± 4,56 điểm phần trăm khả thi, biến động lớn nhất. Đây là mốc ngẫu nhiên, không phải lựa chọn vận hành đáng tin cậy."],
  ["Giới hạn", "Kết luận chỉ áp dụng cho test split Solomon, bộ dựng tuyến tuần tự và ngân sách GP nhỏ. Không có BKS nên không tính optimality gap; chưa áp dụng local search."],
];
analysis.getRange("A5:B11").values = analysisRows;
analysis.getRange("A5:B5").format = { fill: colors.navy, font: { name: fontName, size: 10, bold: true, color: colors.white } };
analysis.getRange("A6:A11").format.font = { name: fontName, size: 10, bold: true, color: colors.navy };
analysis.getRange("B6:B11").format.wrapText = true;
analysis.getRange("A2:B11").format.font = { name: fontName, size: 10, color: "#222222" };
analysis.getRange("A2").format.font = { name: fontName, size: 16, bold: true, color: colors.navy };
analysis.getRange("A:A").format.columnWidth = 18;
analysis.getRange("B:B").format.columnWidth = 105;
analysis.getRange("A6:B11").format.rowHeight = 48;
analysis.getRange("A5:B5").format = { fill: colors.navy, font: { name: fontName, size: 10, bold: true, color: colors.white } };

workbook.recalculate();
const summaryCheck = await workbook.inspect({ kind: "table", range: "Tổng quan!A2:M24", include: "values,formulas", tableMaxRows: 24, tableMaxCols: 13 });
const errorCheck = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan" });
console.log(summaryCheck.ndjson);
console.log(errorCheck.ndjson);

await fs.mkdir(path.dirname(outputPath), { recursive: true });
for (const [sheetName, range] of [
  ["Tổng quan", "A1:M24"], ["Theo lượt", "A1:J20"], ["Chi tiết", "A1:R18"],
  ["Tuyến xe", "A1:L16"], ["GP", "A1:L12"], ["Dữ liệu", "A1:J16"], ["Phân tích", "A1:B11"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(path.join(path.dirname(outputPath), `preview_${sheetName.replaceAll(" ", "_")}.png`), new Uint8Array(await preview.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(`SAVED ${outputPath}`);
