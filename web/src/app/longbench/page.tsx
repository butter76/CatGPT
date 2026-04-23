"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Gauge,
  Loader2,
  PlayCircle,
  CheckCircle2,
  XCircle,
  Hourglass,
  RefreshCw,
  Play,
} from "lucide-react";
import type { BenchmarkRun, BenchmarkRunStatus } from "@/lib/types";

const DEFAULT_MAX_NODES = 1_000_000;

function statusBadge(status: BenchmarkRunStatus) {
  switch (status) {
    case "pending":
      return (
        <Badge variant="outline" className="border-yellow-500 text-yellow-600">
          <Hourglass className="w-3 h-3 mr-1" /> pending
        </Badge>
      );
    case "running":
      return (
        <Badge variant="outline" className="border-blue-500 text-blue-600">
          <Loader2 className="w-3 h-3 mr-1 animate-spin" /> running
        </Badge>
      );
    case "completed":
      return (
        <Badge variant="outline" className="border-green-500 text-green-600">
          <CheckCircle2 className="w-3 h-3 mr-1" /> completed
        </Badge>
      );
    case "failed":
      return (
        <Badge variant="outline" className="border-red-500 text-red-600">
          <XCircle className="w-3 h-3 mr-1" /> failed
        </Badge>
      );
    case "cancelled":
      return (
        <Badge variant="outline" className="border-gray-400 text-gray-500">
          cancelled
        </Badge>
      );
  }
}

function formatDuration(startedAt: string | null, finishedAt: string | null): string {
  if (!startedAt) return "–";
  const start = new Date(startedAt).getTime();
  const end = finishedAt ? new Date(finishedAt).getTime() : Date.now();
  const s = Math.max(0, Math.floor((end - start) / 1000));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${ss}s`;
  return `${ss}s`;
}

export default function LongBenchPage() {
  const router = useRouter();
  const [runs, setRuns] = useState<BenchmarkRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);

  const load = useCallback(() => {
    setLoading(true);
    fetch("/api/longbench/runs")
      .then((r) => r.json())
      .then((data: BenchmarkRun[]) => setRuns(data))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // Auto-refresh while any run is still running.
  useEffect(() => {
    const hasActive = runs.some(
      (r) => r.status === "running" || r.status === "pending"
    );
    if (!hasActive) return;
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, [runs, load]);

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Gauge className="w-6 h-6 text-indigo-500" />
            LongBench
          </h1>
          <p className="text-sm text-muted-foreground">
            Evaluate the network on LongBench positions using Fractional MCTS.
            Each position is scored by how quickly the engine locks in on the
            correct prediction.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={load} disabled={loading}>
            <RefreshCw className={`w-4 h-4 mr-1.5 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </Button>
          <NewRunDialog
            open={dialogOpen}
            onOpenChange={setDialogOpen}
            onCreated={(runId) => {
              setDialogOpen(false);
              router.push(`/longbench/${runId}?autostart=1`);
            }}
          />
        </div>
      </div>

      {loading && runs.length === 0 ? (
        <div className="flex justify-center py-16">
          <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
        </div>
      ) : runs.length === 0 ? (
        <Card>
          <CardContent className="py-16 text-center text-muted-foreground space-y-2">
            <Gauge className="w-10 h-10 mx-auto text-indigo-500/60" />
            <p className="text-lg font-medium">No benchmark runs yet</p>
            <p className="text-sm">
              Flag a few positions as LongBench in the Position Database, then
              start a run above.
            </p>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="border-b bg-muted/30">
                  <tr className="text-left text-xs uppercase tracking-wide text-muted-foreground">
                    <th className="py-2 px-4">Run</th>
                    <th className="py-2 px-4">Status</th>
                    <th className="py-2 px-4">Engine</th>
                    <th className="py-2 px-4 text-right">Max Nodes</th>
                    <th className="py-2 px-4 text-right">Positions</th>
                    <th className="py-2 px-4 text-right">Score</th>
                    <th className="py-2 px-4 text-right">Duration</th>
                    <th className="py-2 px-4">Started</th>
                    <th className="py-2 px-4" />
                  </tr>
                </thead>
                <tbody>
                  {runs.map((run) => (
                    <tr
                      key={run.id}
                      className="border-b last:border-b-0 hover:bg-muted/30 transition-colors"
                    >
                      <td className="py-2 px-4 font-mono text-xs">
                        <Link
                          href={`/longbench/${run.id}`}
                          className="hover:underline text-indigo-500"
                        >
                          #{run.id}
                        </Link>
                      </td>
                      <td className="py-2 px-4">{statusBadge(run.status)}</td>
                      <td className="py-2 px-4 max-w-xs truncate font-mono text-xs text-muted-foreground">
                        {run.engine}
                      </td>
                      <td className="py-2 px-4 text-right font-mono">
                        {run.maxNodes.toLocaleString()}
                      </td>
                      <td className="py-2 px-4 text-right font-mono">
                        {run.positionCount ?? "–"}
                      </td>
                      <td className="py-2 px-4 text-right font-mono">
                        {run.aggregateScore != null
                          ? run.aggregateScore.toFixed(3)
                          : "–"}
                      </td>
                      <td className="py-2 px-4 text-right font-mono text-muted-foreground">
                        {formatDuration(run.startedAt, run.finishedAt)}
                      </td>
                      <td className="py-2 px-4 text-xs text-muted-foreground">
                        {run.startedAt
                          ? new Date(run.startedAt).toLocaleString()
                          : new Date(run.createdAt).toLocaleString()}
                      </td>
                      <td className="py-2 px-4 text-right">
                        <Button
                          asChild
                          variant="ghost"
                          size="sm"
                          className="h-7 text-xs"
                        >
                          <Link href={`/longbench/${run.id}`}>Open</Link>
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function NewRunDialog({
  open,
  onOpenChange,
  onCreated,
}: {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  onCreated: (runId: number) => void;
}) {
  const [engine, setEngine] = useState("");
  const [maxNodes, setMaxNodes] = useState(String(DEFAULT_MAX_NODES));
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const body: { engine?: string; maxNodes?: number } = {};
      if (engine.trim()) body.engine = engine.trim();
      const parsed = Number(maxNodes);
      if (Number.isFinite(parsed) && parsed > 0) body.maxNodes = Math.floor(parsed);
      const res = await fetch("/api/longbench/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const { error: msg } = await res.json().catch(() => ({ error: res.statusText }));
        throw new Error(msg || "Failed to start run");
      }
      const run = (await res.json()) as BenchmarkRun;
      onCreated(run.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogTrigger asChild>
        <Button size="sm">
          <Play className="w-4 h-4 mr-1.5" />
          Start new run
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Start a LongBench run</DialogTitle>
          <DialogDescription>
            The run will iterate every position flagged as LongBench, searching
            up to the specified node budget with Fractional MCTS.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="space-y-1.5">
            <Label htmlFor="engine">Engine label</Label>
            <Input
              id="engine"
              value={engine}
              onChange={(e) => setEngine(e.target.value)}
              placeholder="(defaults to CATGPT_ENGINE_PATH)"
            />
            <p className="text-xs text-muted-foreground">
              Used for identifying this run. The actual network path is picked
              up from the server&apos;s <code>CATGPT_ENGINE_PATH</code> env var.
            </p>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="maxNodes">Max nodes per position</Label>
            <Input
              id="maxNodes"
              value={maxNodes}
              onChange={(e) => setMaxNodes(e.target.value)}
              inputMode="numeric"
            />
          </div>
          {error && (
            <p className="text-sm text-red-500">{error}</p>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={submitting}>
            Cancel
          </Button>
          <Button onClick={submit} disabled={submitting}>
            {submitting ? (
              <Loader2 className="w-4 h-4 mr-1.5 animate-spin" />
            ) : (
              <PlayCircle className="w-4 h-4 mr-1.5" />
            )}
            Start
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
