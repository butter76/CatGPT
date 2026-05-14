"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Button } from "@/components/ui/button";
import { usePositionStore } from "@/lib/store";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { NotationFormat } from "@/lib/types";
import { Database, FlaskConical, Cat, Gauge } from "lucide-react";

const NAV_ITEMS = [
  { href: "/positions", label: "Position Database", icon: Database },
  { href: "/longbench", label: "LongBench", icon: Gauge },
  { href: "/analyze", label: "Quick Analysis", icon: FlaskConical },
];

export function Navbar() {
  const pathname = usePathname();
  const { notationFormat, setNotationFormat } = usePositionStore();

  return (
    <header className="sticky top-0 z-50 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex h-14 items-center justify-between">
          {/* Logo + Nav */}
          <div className="flex items-center gap-6">
            <Link
              href="/"
              className="flex items-center gap-2 font-bold text-lg hover:opacity-80 transition-opacity"
            >
              <Cat className="w-6 h-6 text-primary" />
              <span>CatGPT Analysis</span>
            </Link>

            <nav className="hidden sm:flex items-center gap-1">
              {NAV_ITEMS.map((item) => {
                const Icon = item.icon;
                const isActive =
                  pathname === item.href || pathname.startsWith(item.href + "/");
                return (
                  <Link key={item.href} href={item.href}>
                    <Button
                      variant={isActive ? "secondary" : "ghost"}
                      size="sm"
                      className="gap-1.5"
                    >
                      <Icon className="w-4 h-4" />
                      {item.label}
                    </Button>
                  </Link>
                );
              })}
            </nav>
          </div>

          {/* Settings */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground hidden sm:inline">
                Notation:
              </span>
              <Select
                value={notationFormat}
                onValueChange={(v) => setNotationFormat(v as NotationFormat)}
              >
                <SelectTrigger className="w-[120px] h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="algebraic">Algebraic</SelectItem>
                  <SelectItem value="uci">UCI</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
