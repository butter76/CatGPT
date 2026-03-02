"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { fetchPositions } from "@/lib/store";
import type { Position } from "@/lib/types";
import { Database, FlaskConical, Zap, Castle, Loader2 } from "lucide-react";

export default function HomePage() {
  const [positions, setPositions] = useState<Position[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchPositions()
      .then(setPositions)
      .finally(() => setLoading(false));
  }, []);

  const sharpCount = positions.filter((p) => p.type === "SHARP").length;
  const fortressCount = positions.filter((p) => p.type === "FORTRESS").length;

  return (
    <div className="space-y-8">
      {/* Hero */}
      <div className="text-center space-y-3 py-8">
        <h1 className="text-4xl font-bold tracking-tight">
          CatGPT Position Analysis
        </h1>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          Analyze chess positions with CatGPT&apos;s neural network. Evaluate sharp tactical
          positions and fortress endgames. Build a growing database of critical positions.
        </p>
      </div>

      {/* Stats Cards */}
      {loading ? (
        <div className="flex justify-center py-8">
          <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Total Positions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2">
                <Database className="w-5 h-5 text-primary" />
                <span className="text-3xl font-bold">{positions.length}</span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Sharp Positions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2">
                <Zap className="w-5 h-5 text-amber-500" />
                <span className="text-3xl font-bold">{sharpCount}</span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Fortress Positions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2">
                <Castle className="w-5 h-5 text-blue-500" />
                <span className="text-3xl font-bold">{fortressCount}</span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Analyzed
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2">
                <FlaskConical className="w-5 h-5 text-green-500" />
                <span className="text-3xl font-bold">
                  {positions.filter((p) => p.networkAnalysis).length}
                </span>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-amber-500/10 to-transparent" />
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="w-5 h-5" />
              Position Database
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-sm text-muted-foreground">
              Browse and manage your collection of SHARP and FORTRESS positions.
              View network analysis, policy distributions, and move annotations.
            </p>
            <Link href="/positions">
              <Button>Browse Positions →</Button>
            </Link>
          </CardContent>
        </Card>

        <Card className="relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 to-transparent" />
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FlaskConical className="w-5 h-5" />
              Quick Analysis
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-sm text-muted-foreground">
              Paste a FEN string and instantly see the board. Request analysis
              from CatGPT&apos;s network, then optionally save it to the database.
            </p>
            <Link href="/analyze">
              <Button variant="secondary">Analyze Position →</Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
