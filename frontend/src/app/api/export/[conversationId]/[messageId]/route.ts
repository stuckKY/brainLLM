import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ conversationId: string; messageId: string }> }
) {
  const { conversationId, messageId } = await params;
  const backendUrl = `${BACKEND_URL}/conversations/${conversationId}/messages/${messageId}/export?format=pdf`;
  console.log(
    `[proxy] GET /api/export/${conversationId}/${messageId} → ${backendUrl}`
  );

  try {
    const res = await fetch(backendUrl);

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      console.error(`[proxy] Backend returned ${res.status}:`, data);
      return NextResponse.json(data, { status: res.status });
    }

    const pdfBuffer = await res.arrayBuffer();
    const contentDisposition =
      res.headers.get("Content-Disposition") ||
      'attachment; filename="export.pdf"';

    return new Response(pdfBuffer, {
      status: 200,
      headers: {
        "Content-Type": "application/pdf",
        "Content-Disposition": contentDisposition,
      },
    });
  } catch (err) {
    console.error("[proxy] Failed to reach backend:", err);
    return NextResponse.json(
      { detail: `Failed to reach backend at ${BACKEND_URL}: ${err}` },
      { status: 502 }
    );
  }
}
