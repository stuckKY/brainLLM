import { NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function GET() {
  console.log(`[proxy] GET /api/files → ${BACKEND_URL}/files`);

  try {
    const res = await fetch(`${BACKEND_URL}/files`);
    const data = await res.json();

    if (!res.ok) {
      console.error(`[proxy] Backend returned ${res.status}:`, data);
      return NextResponse.json(data, { status: res.status });
    }

    console.log("[proxy] Files listed:", data.count);
    return NextResponse.json(data);
  } catch (err) {
    console.error("[proxy] Failed to reach backend:", err);
    return NextResponse.json(
      { detail: `Failed to reach backend at ${BACKEND_URL}: ${err}` },
      { status: 502 }
    );
  }
}
