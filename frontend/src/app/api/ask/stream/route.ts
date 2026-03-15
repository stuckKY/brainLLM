import { NextRequest } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function POST(request: NextRequest) {
  console.log(`[proxy] POST /api/ask/stream → ${BACKEND_URL}/ask/stream`);

  try {
    const body = await request.json();
    console.log("[proxy] Stream request body:", JSON.stringify(body));

    const res = await fetch(`${BACKEND_URL}/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      console.error(`[proxy] Backend returned ${res.status}:`, data);
      return new Response(JSON.stringify(data), {
        status: res.status,
        headers: { "Content-Type": "application/json" },
      });
    }

    // Forward the SSE stream directly — no buffering
    return new Response(res.body, {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
      },
    });
  } catch (err) {
    console.error("[proxy] Failed to reach backend:", err);
    return new Response(
      JSON.stringify({
        detail: `Failed to reach backend at ${BACKEND_URL}: ${err}`,
      }),
      { status: 502, headers: { "Content-Type": "application/json" } }
    );
  }
}
