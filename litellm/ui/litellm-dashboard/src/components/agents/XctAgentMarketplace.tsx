import { useQuery } from "@tanstack/react-query";
import { createApiClient } from "@/lib/http/client";
import { useState } from "react";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { toast } from "@/lib/toast";
import { createAgentCall } from "../networking";
import { useTranslation } from "react-i18next";

const agentFields = {
  slug: z.string(),
  name: z.string(),
  description: z.string(),
  category: z.string(),
  emoji: z.string().default(""),
};
const agentSchema = z.object(agentFields);
type Agent = z.infer<typeof agentSchema>;
const marketplaceClient = createApiClient({ getBaseUrl: () => "" });
const DEFAULT_GATEWAY_URL = "https://xct-agents-production.up.railway.app";
interface Props {
  accessToken: string | null;
  isAdmin: boolean;
  gatewayUrl?: string;
  onAgentAdded?: () => void;
}
function useMarketplaceData(accessToken: string | null, providedUrl: string | undefined, slug: string | undefined) {
  const configQuery = useQuery({
    queryKey: ["xct-marketplace-config", accessToken],
    enabled: !providedUrl && Boolean(accessToken),
    queryFn: async ({ signal }) => {
      const response = await marketplaceClient.get<unknown>("/v1/xct-marketplace/config", { accessToken, signal });
      return z.object({ gateway_url: z.string().url() }).parse(response);
    },
  });
  const gatewayUrl = providedUrl || configQuery.data?.gateway_url || DEFAULT_GATEWAY_URL;
  const agentsQuery = useQuery({
    queryKey: ["xct-marketplace-agents", gatewayUrl],
    queryFn: async ({ signal }) => {
      const response = await marketplaceClient.get<unknown>(`${gatewayUrl.replace(/\/$/, "")}/agents`, { signal });
      return z.object({ agents: z.array(agentSchema) }).parse(response).agents;
    },
  });
  const detailQuery = useQuery({
    queryKey: ["xct-marketplace-detail", gatewayUrl, slug],
    enabled: Boolean(slug),
    queryFn: async ({ signal }) => {
      const response = await marketplaceClient.get<unknown>(
        `${gatewayUrl.replace(/\/$/, "")}/agents/${encodeURIComponent(slug ?? "")}/`,
        { signal },
      );
      return z.record(z.string(), z.unknown()).parse(response);
    },
  });
  return { agentsQuery, detailQuery, gatewayUrl };
}

function detailDescription(detail: Record<string, unknown> | undefined): string {
  return typeof detail?.description === "string" ? detail.description.slice(0, 500) : "";
}

export default function XCTAgentMarketplace({ accessToken, isAdmin, gatewayUrl: providedUrl, onAgentAdded }: Props) {
  const { t } = useTranslation();
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("all");
  const [selected, setSelected] = useState<Agent | null>(null);
  const [importing, setImporting] = useState(false);
  const { agentsQuery, detailQuery, gatewayUrl } = useMarketplaceData(accessToken, providedUrl, selected?.slug);
  const agents = agentsQuery.data ?? [];
  const loading = agentsQuery.isFetching;
  const error = agentsQuery.isError ? t("agents.marketplace.loadFailed") : "";
  const detail = detailQuery.data;
  const detailLoading = detailQuery.isFetching;
  const importAgent = async () => {
    const cannotImport = !isAdmin || importing;
    if (cannotImport) return;
    if (!selected || !accessToken || !detail) return;
    setImporting(true);
    try {
      await createAgentCall(accessToken, {
        agent_name: `xct-${selected.slug}`,
        litellm_params: { model: `a2a/${selected.slug}` },
        agent_card_params: {
          ...detail,
          name: selected.name,
          description: selected.description,
          url: `${gatewayUrl.replace(/\/$/, "")}/agents/${encodeURIComponent(selected.slug)}/`,
        },
      });
      toast.success(t("agents.marketplace.imported", { name: selected.name }));
      setSelected(null);
      onAgentAdded?.();
    } catch {
      toast.error(t("agents.marketplace.importFailed"));
    } finally {
      setImporting(false);
    }
  };
  const visible = agents.filter(
    (agent) =>
      (category === "all" || agent.category === category) &&
      `${agent.name} ${agent.description} ${agent.category}`.toLowerCase().includes(query.toLowerCase()),
  );
  return (
    <section className="space-y-4 rounded-lg border p-4">
      <h2 className="text-xl font-semibold">{t("agents.marketplace.title")}</h2>
      <p>{t("agents.marketplace.description")}</p>
      <div className="flex flex-wrap items-end gap-3">
        <label className="grid gap-1">
          {t("agents.marketplace.search")}
          <Input value={query} onChange={(event) => setQuery(event.target.value)} />
        </label>
        <label className="grid gap-1">
          {t("agents.marketplace.category")}
          <select
            className="rounded border bg-background p-2"
            value={category}
            onChange={(event) => setCategory(event.target.value)}
          >
            <option value="all">{t("agents.marketplace.allCategories")}</option>
            {[...new Set(agents.map((agent) => agent.category))].sort().map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
        </label>
        <Button variant="outline" onClick={() => void agentsQuery.refetch()} disabled={loading}>
          {t("agents.marketplace.refresh")}
        </Button>
      </div>
      {error && <p role="alert">{error}</p>}
      {loading && <p role="status">{t("agents.marketplace.loading")}</p>}
      <div className="grid max-h-[600px] gap-4 overflow-y-auto sm:grid-cols-2 lg:grid-cols-3">
        {visible.map((agent) => (
          <button
            type="button"
            key={agent.slug}
            onClick={() => setSelected(agent)}
            className="space-y-2 rounded-lg border p-4 text-left hover:bg-muted"
          >
            <h3 className="font-semibold">
              {agent.emoji} {agent.name}
            </h3>
            <p className="line-clamp-3">{agent.description}</p>
            <span>{agent.category}</span>
          </button>
        ))}
      </div>
      <p>{t("agents.marketplace.showing", { visible: visible.length, total: agents.length })}</p>
      {!loading && !visible.length && <p>{t("agents.marketplace.noMatches")}</p>}
      <Dialog
        open={selected !== null}
        onOpenChange={(open) => {
          if (!open && !importing) setSelected(null);
        }}
      >
        <DialogContent className="max-h-[85vh] overflow-y-auto sm:max-w-2xl">
          <DialogTitle>{selected?.name ?? t("agents.marketplace.agentFallback")}</DialogTitle>
          <p>{selected?.description}</p>
          <p>{t("agents.marketplace.categoryValue", { category: selected?.category })}</p>
          <code>xct-{selected?.slug}</code>
          {detailQuery.isError && <p role="alert">{t("agents.marketplace.detailsFailed")}</p>}
          {detailLoading ? (
            <p role="status">{t("agents.marketplace.detailsLoading")}</p>
          ) : (
            <pre className="max-h-64 overflow-y-auto whitespace-pre-wrap">{detailDescription(detail)}</pre>
          )}
          {isAdmin ? (
            <Button disabled={importing || !detail || detailLoading} onClick={importAgent}>
              {importing ? t("agents.marketplace.importing") : t("agents.marketplace.add")}
            </Button>
          ) : (
            <p>{t("agents.marketplace.adminOnly")}</p>
          )}
        </DialogContent>
      </Dialog>
    </section>
  );
}
