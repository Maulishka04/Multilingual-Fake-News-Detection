import { toast } from "react-hot-toast";
import { ERROR_MESSAGES } from "./constants";

const downloadBlob = (filename: string, content: string, contentType: string): void => {
  const blob = new Blob([content], { type: contentType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
};

export const exportToJson = <T>(filename: string, data: T): void => {
  downloadBlob(filename, JSON.stringify(data, null, 2), "application/json;charset=utf-8;");
};

export const exportToCsv = (filename: string, rows: Array<Record<string, string | number | boolean>>): void => {
  if (!rows.length) {
    downloadBlob(filename, "", "text/csv;charset=utf-8;");
    return;
  }

  const headers = Object.keys(rows[0]);
  const csvRows = rows.map((row) =>
    headers
      .map((header) => {
        const rawValue = String(row[header] ?? "");
        return `"${rawValue.replace(/\"/g, "\"\"")}"`;
      })
      .join(",")
  );

  const csv = [headers.join(","), ...csvRows].join("\n");
  downloadBlob(filename, csv, "text/csv;charset=utf-8;");
};

export const copyTextToClipboard = async (value: string): Promise<void> => {
  try {
    await navigator.clipboard.writeText(value);
  } catch {
    toast.error(ERROR_MESSAGES.unknown);
    throw new Error("Clipboard write failed");
  }
};
