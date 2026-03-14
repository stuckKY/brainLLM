import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function POST(request: NextRequest) {
  console.log(`[proxy] POST /api/ask → ${BACKEND_URL}/ask`);

  try {
    const body = await request.json();
    console.log("[proxy] Request body:", JSON.stringify(body));

    const res = await fetch(`${BACKEND_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();

    if (!res.ok) {
      console.error(`[proxy] Backend returned ${res.status}:`, data);
      return NextResponse.json(data, { status: res.status });
    }

    console.log("[proxy] Ask success, chunks_used:", data.chunks_used);
    return NextResponse.json(data);
  } catch (err) {
    console.error("[proxy] Failed to reach backend:", err);
    return NextResponse.json(
      { detail: `Failed to reach backend at ${BACKEND_URL}: ${err}` },
      { status: 502 }
    );
  }
}
