"use client";

import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import XCTAgentMarketplace from "@/components/agents/XctAgentMarketplace";
import { isAdminRole } from "@/utils/roles";
import AgentsPanel from "./_components/AgentsPanel";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { useTranslation } from "react-i18next";

export default function Agents() {
  const { t } = useTranslation();
  const [revision, setRevision] = useState(0);
  const { accessToken, userRole } = useAuthorized();
  const { data: teams } = useTeams();
  return (
    <Tabs defaultValue="agents">
      <TabsList>
        <TabsTrigger value="agents">{t("agents.marketplace.tabMyAgents")}</TabsTrigger>
        <TabsTrigger value="marketplace">{t("agents.marketplace.tabMarketplace")}</TabsTrigger>
      </TabsList>
      <TabsContent value="agents">
        <AgentsPanel key={revision} accessToken={accessToken} userRole={userRole} teams={teams ?? null} />
      </TabsContent>
      <TabsContent value="marketplace">
        <XCTAgentMarketplace
          accessToken={accessToken}
          isAdmin={userRole ? isAdminRole(userRole) : false}
          onAgentAdded={() => setRevision((value) => value + 1)}
        />
      </TabsContent>
    </Tabs>
  );
}
