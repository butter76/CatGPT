"use client";

import { useState, useMemo, useEffect, useCallback } from "react";
import Link from "next/link";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CompactBoard } from "@/components/chess/analysis-board";
import { AddPositionDialog } from "@/components/chess/add-position-dialog";
import { fetchPositions } from "@/lib/store";
import type { Position, PositionType, Outcome, BlunderTag } from "@/lib/types";
import {
  Plus,
  Search,
  Zap,
  Castle,
  Trophy,
  Minus,
  Skull,
  HelpCircle,
  ArrowRight,
  Loader2,
  Tag,
  X,
  Gauge,
} from "lucide-react";

type FilterType = "ALL" | PositionType;
type BlunderFilterType = "ALL" | BlunderTag;

const BLUNDER_TAG_CONFIG: Record<BlunderTag, { label: string; color: string }> = {
  catgpt: { label: "🐱 CatGPT", color: "border-orange-500 text-orange-600 bg-orange-50" },
  stockfish: { label: "🐟 Stockfish", color: "border-green-500 text-green-600 bg-green-50" },
  leela: { label: "♟️ Leela", color: "border-purple-500 text-purple-600 bg-purple-50" },
};

const OUTCOME_ICONS: Record<Outcome, React.ReactNode> = {
  win: <Trophy className="w-3 h-3 text-green-500" />,
  loss: <Skull className="w-3 h-3 text-red-500" />,
  draw: <Minus className="w-3 h-3 text-gray-400" />,
  unknown: <HelpCircle className="w-3 h-3 text-yellow-500" />,
};

export default function PositionsPage() {
  const [positions, setPositions] = useState<Position[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<FilterType>("ALL");
  const [blunderFilter, setBlunderFilter] = useState<BlunderFilterType>("ALL");
  const [longBenchOnly, setLongBenchOnly] = useState(false);
  const [search, setSearch] = useState("");

  const loadPositions = useCallback(() => {
    setLoading(true);
    fetchPositions()
      .then(setPositions)
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    loadPositions();
  }, [loadPositions]);

  const filtered = useMemo(() => {
    let result = positions;
    if (filter !== "ALL") {
      result = result.filter((p) => p.type === filter);
    }
    if (blunderFilter !== "ALL") {
      result = result.filter((p) => p.blunderTag === blunderFilter);
    }
    if (longBenchOnly) {
      result = result.filter((p) => p.longBench);
    }
    if (search.trim()) {
      const q = search.toLowerCase();
      result = result.filter(
        (p) =>
          p.name.toLowerCase().includes(q) ||
          p.description?.toLowerCase().includes(q) ||
          p.fen.toLowerCase().includes(q)
      );
    }
    return result;
  }, [positions, filter, blunderFilter, longBenchOnly, search]);

  const longBenchCount = useMemo(
    () => positions.filter((p) => p.longBench).length,
    [positions]
  );

  // Count positions by blunder tag for filter badges
  const blunderCounts = useMemo(() => {
    const counts: Record<BlunderTag | "tagged", number> = { catgpt: 0, stockfish: 0, leela: 0, tagged: 0 };
    for (const p of positions) {
      if (p.blunderTag) {
        counts[p.blunderTag]++;
        counts.tagged++;
      }
    }
    return counts;
  }, [positions]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">Position Database</h1>
          <p className="text-sm text-muted-foreground">
            {positions.length} position{positions.length !== 1 && "s"} in the
            database
          </p>
        </div>
        <AddPositionDialog onAdded={loadPositions}>
          <Button>
            <Plus className="w-4 h-4 mr-1.5" />
            Add Position
          </Button>
        </AddPositionDialog>
      </div>

      {/* Filters */}
      <div className="flex flex-col gap-3">
        <div className="flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1 max-w-sm">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search positions..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-9"
            />
          </div>
          <Tabs
            value={filter}
            onValueChange={(v) => setFilter(v as FilterType)}
          >
            <TabsList>
              <TabsTrigger value="ALL">All</TabsTrigger>
              <TabsTrigger value="SHARP" className="gap-1">
                <Zap className="w-3.5 h-3.5" /> Sharp
              </TabsTrigger>
              <TabsTrigger value="FORTRESS" className="gap-1">
                <Castle className="w-3.5 h-3.5" /> Fortress
              </TabsTrigger>
            </TabsList>
          </Tabs>
          <Button
            variant={longBenchOnly ? "secondary" : "outline"}
            size="sm"
            className={`h-9 gap-1.5 ${
              longBenchOnly
                ? "border-indigo-500 text-indigo-600 bg-indigo-500/10"
                : ""
            }`}
            onClick={() => setLongBenchOnly((v) => !v)}
            disabled={longBenchCount === 0 && !longBenchOnly}
            title={
              longBenchCount === 0
                ? "No positions are in LongBench yet"
                : undefined
            }
          >
            <Gauge className="w-4 h-4" />
            LongBench
            <span className="text-xs opacity-70">({longBenchCount})</span>
            {longBenchOnly && <X className="w-3 h-3 ml-0.5" />}
          </Button>
        </div>

        {/* Blunder Tag Filter */}
        {blunderCounts.tagged > 0 && (
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs text-muted-foreground flex items-center gap-1">
              <Tag className="w-3 h-3" /> Blunder tags:
            </span>
            <Button
              variant={blunderFilter === "ALL" ? "secondary" : "ghost"}
              size="sm"
              className="h-7 text-xs"
              onClick={() => setBlunderFilter("ALL")}
            >
              All ({positions.length})
            </Button>
            {(Object.keys(BLUNDER_TAG_CONFIG) as BlunderTag[]).map((tag) =>
              blunderCounts[tag] > 0 ? (
                <Button
                  key={tag}
                  variant={blunderFilter === tag ? "secondary" : "ghost"}
                  size="sm"
                  className={`h-7 text-xs ${
                    blunderFilter === tag ? BLUNDER_TAG_CONFIG[tag].color : ""
                  }`}
                  onClick={() => setBlunderFilter(blunderFilter === tag ? "ALL" : tag)}
                >
                  {BLUNDER_TAG_CONFIG[tag].label} ({blunderCounts[tag]})
                  {blunderFilter === tag && (
                    <X className="w-3 h-3 ml-1" />
                  )}
                </Button>
              ) : null
            )}
          </div>
        )}
      </div>

      {/* Loading */}
      {loading && (
        <div className="flex justify-center py-16">
          <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
        </div>
      )}

      {/* Position Grid */}
      {!loading && filtered.length === 0 ? (
        <div className="text-center py-16 text-muted-foreground">
          <p className="text-lg">No positions found</p>
          <p className="text-sm mt-1">
            {search
              ? "Try a different search term"
              : "Add your first position to get started"}
          </p>
        </div>
      ) : (
        !loading && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filtered.map((position) => (
              <Link
                key={position.id}
                href={`/positions/${position.id}`}
                className="group"
              >
                <Card className="h-full transition-all hover:shadow-lg hover:border-primary/30 group-hover:-translate-y-0.5">
                  <CardContent className="p-4 space-y-3">
                    {/* Board preview */}
                    <div className="flex justify-center">
                      <CompactBoard fen={position.fen} width={180} />
                    </div>

                    {/* Info */}
                    <div className="space-y-2">
                      <div className="flex items-start justify-between gap-2">
                        <h3 className="font-semibold text-sm leading-tight line-clamp-1">
                          {position.name}
                        </h3>
                        <Badge
                          variant="outline"
                          className={`shrink-0 text-[10px] ${
                            position.type === "SHARP"
                              ? "border-amber-500 text-amber-600"
                              : "border-blue-500 text-blue-600"
                          }`}
                        >
                          {position.type === "SHARP" ? (
                            <Zap className="w-3 h-3 mr-0.5" />
                          ) : (
                            <Castle className="w-3 h-3 mr-0.5" />
                          )}
                          {position.type}
                        </Badge>
                      </div>

                      {position.description && (
                        <p className="text-xs text-muted-foreground line-clamp-2">
                          {position.description}
                        </p>
                      )}

                      {/* Type-specific info */}
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2 text-xs text-muted-foreground">
                          {position.type === "SHARP" &&
                            position.moveAnnotations && (
                              <span>
                                {
                                  position.moveAnnotations.filter(
                                    (a) => a.annotation === "correct"
                                  ).length
                                }{" "}
                                correct,{" "}
                                {
                                  position.moveAnnotations.filter(
                                    (a) => a.annotation === "blunder"
                                  ).length
                                }{" "}
                                blunders
                              </span>
                            )}
                          {position.type === "FORTRESS" &&
                            position.expectedOutcome && (
                              <span className="flex items-center gap-1">
                                {OUTCOME_ICONS[position.expectedOutcome]}
                                {position.expectedOutcome === "draw"
                                  ? "Drawn"
                                  : position.expectedOutcome === "win"
                                  ? "Decisive (Win)"
                                  : position.expectedOutcome === "unknown"
                                  ? "Unknown"
                                  : "Decisive (Loss)"}
                              </span>
                            )}
                        </div>
                        <ArrowRight className="w-4 h-4 text-muted-foreground group-hover:text-primary transition-colors" />
                      </div>

                      {/* Blunder tag + LongBench */}
                      <div className="flex items-center gap-1.5 flex-wrap">
                        {position.blunderTag && (
                          <Badge
                            variant="outline"
                            className={`text-[10px] ${BLUNDER_TAG_CONFIG[position.blunderTag].color}`}
                          >
                            {BLUNDER_TAG_CONFIG[position.blunderTag].label} Blunder
                          </Badge>
                        )}
                        {position.longBench && (
                          <Badge
                            variant="outline"
                            className="text-[10px] border-indigo-500 text-indigo-600 bg-indigo-500/10"
                          >
                            <Gauge className="w-2.5 h-2.5 mr-0.5" />
                            LongBench
                          </Badge>
                        )}
                      </div>

                      {/* Analysis indicator */}
                      {position.networkAnalysis && (
                        <div className="flex items-center gap-1">
                          <div className="w-1.5 h-1.5 rounded-full bg-green-500" />
                          <span className="text-[10px] text-muted-foreground">
                            Network analysis available
                          </span>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </Link>
            ))}
          </div>
        )
      )}
    </div>
  );
}
