import { neon, neonConfig } from "@neondatabase/serverless";
import { drizzle } from "drizzle-orm/neon-http";
import * as schema from "./schema";

const connectionString = process.env.DATABASE_URL!;

// When pointing at the local Postgres + neon-proxy stack (see web/docker-compose.yml),
// route the serverless driver's HTTP queries to the local proxy on port 4444.
const localProxyHost = new URL(connectionString).hostname;
if (localProxyHost === "db.localtest.me" || localProxyHost === "localhost") {
  neonConfig.fetchEndpoint = (host) => `http://${host}:4444/sql`;
}

const sql = neon(connectionString);
export const db = drizzle(sql, { schema });
