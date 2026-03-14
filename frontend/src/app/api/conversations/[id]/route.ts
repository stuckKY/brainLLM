import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  console.log(
    `[proxy] GET /api/conversations/${id} → ${BACKEND_URL}/conversations/${id}`
  );

  try {
    const res = await fetch(`${BACKEND_URL}/conversations/${id}`);
    const data = await res.json();

    if (!res.ok) {
      console.error(`[proxy] Backend returned ${res.status}:`, data);
      return NextResponse.json(data, { status: res.status });
    }

    console.log("[proxy] Conversation loaded:", data.messages?.length, "messages");
    return NextResponse.json(data);
  } catch (err) {
    console.error("[proxy] Failed to reach backend:", err);
    return NextResponse.json(
      { detail: `Failed to reach backend at ${BACKEND_URL}: ${err}` },
      { status: 502 }
    );
  }
}

export async function DELETE(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  console.log(
    `[proxy] DELETE /api/conversations/${id} → ${BACKEND_URL}/conversations/${id}`
  );

  try {
    const res = await fetch(`${BACKEND_URL}/conversations/${id}`, {
      method: "DELETE",
    });
    const data = await res.json();

    if (!res.ok) {
      console.error(`[proxy] Backend returned ${res.status}:`, data);
      return NextResponse.json(data, { status: res.status });
    }

    console.log("[proxy] Conversation deleted");
    return NextResponse.json(data);
  } catch (err) {
    console.error("[proxy] Failed to reach backend:", err);
    return NextResponse.json(
      { detail: `Failed to reach backend at ${BACKEND_URL}: ${err}` },
      { status: 502 }
    );
  }
}

export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  console.log(
    `[proxy] PATCH /api/conversations/${id} → ${BACKEND_URL}/conversations/${id}`
  );

  try {
    const body = await request.json();
    const res = await fetch(`${BACKEND_URL}/conversations/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();

    if (!res.ok) {
      console.error(`[proxy] Backend returned ${res.status}:`, data);
      return NextResponse.json(data, { status: res.status });
    }

    console.log("[proxy] Conversation renamed");
    return NextResponse.json(data);
  } catch (err) {
    console.error("[proxy] Failed to reach backend:", err);
    return NextResponse.json(
      { detail: `Failed to reach backend at ${BACKEND_URL}: ${err}` },
      { status: 502 }
    );
  }
}
