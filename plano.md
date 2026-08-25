# Plano causal para generalizar readbacks, GPU authority e fencing

## 1. Decisão de projeto

O mecanismo atual de God of War III deve ser tratado como uma **prova funcional**, não como a
especificação do fast path. Ele demonstra que é possível remover a espera imediata do GCP,
preservar os bytes produzidos pela GPU e materializá-los apenas quando houver demanda real. Ele
não demonstra que `CUSA01715`, 512 bytes, `R16G16Sfloat`, um EOS ou `value == 1` sejam requisitos
semânticos.

O desenho correto será **candidate-first, orientado ao consumidor e fail-closed**:

1. Toda solicitação conservadora de download vira um candidato, sem filtro por título, tamanho,
   formato ou pacote.
2. Os pacotes PM4 são decodificados em um escopo de conclusão/visibilidade. O pacote não decide
   sozinho o destino dos dados.
3. O escopo é cruzado com a linhagem, versão, intervalo, aliases e consumidores do recurso.
4. O avaliador escolhe entre manter a imagem GPU-side, criar um shadow GPU-side, materializar sob
   demanda ou usar o readback legado.
5. Toda rejeição registra todos os motivos aplicáveis. O fallback legado continua sendo a
   resposta para qualquer lacuna de prova.

Arquiteturalmente, GPU authority é o destino normal quando nenhum observador exige host visibility
ou materialização. Funcionalmente, porém, isso só é promovido quando UID/epoch/alias/range/version,
producer, consumer e lifetime estão provados. O avaliador não mantém uma allowlist infinita de
pacotes “seguros”; ele pergunta quais requisitos concretos obrigam sair da GPU e registra cada
lacuna que impede provar a opção mais barata. Melhorar o tracker remove blockers estruturais e
amplia naturalmente o caminho GPU-side, sem transformar ausência de evidência em evidência de
ausência.

O objetivo final não é “remover fences” nem “reduzir a contagem de readbacks” como fins em si. O
objetivo é preservar exatamente as observações que o jogo pode fazer, evitando cópia, submit,
quebra de rendering e espera de host quando essas operações não participam da observação.

## 2. Estado conhecido e o que ele realmente prova

### 2.1 Implementação funcional atual

O caminho funcional está distribuído principalmente nestes pontos:

- `TextureCache::ScheduleRenderTargetDownload`, em
  `src/video_core/texture_cache/texture_cache.cpp`, reconhece somente uma imagem de God of War III
  linear, não-storage, `R16G16Sfloat`, 1x1x1, pitch 128, um mip, uma layer e 512 bytes.
- `TextureCache::ScheduleImageDownload` mantém uma lista real de `PendingImageDownload`, mas o
  fast path usa paralelamente um único `pending_fastpath_candidate`; um candidato posterior pode
  substituir o anterior.
- `Liverpool::ProcessEventWriteEos`, em `src/video_core/amdgpu/liverpool.cpp`, tenta promover esse
  singleton somente para `SignalFence`, valor 1 e label não nulo.
- `TextureCache::PromotePendingDownloadAuthority` exige exatamente uma pendência, valida a imagem,
  grava `copyImageToBuffer`, cria um `GpuAuthorityShadow`, instala um pin e remove a pendência.
- `GpuAuthorityTracker` associa imagem/epoch, shadow, label e timeline tick; publica o label após a
  conclusão física e materializa os bytes somente quando `ResolveForRamRead` encontra demanda.
- `GpuAuthorityTracker::MatchVirtualWait` aceita apenas a assinatura observada de `WAIT_REG_MEM`:
  Equal ou GreaterEqual, referência 1 e máscara completa.
- `BufferCache::ObtainBufferForImage` reutiliza diretamente o shadow apenas para a faixa exata de
  512 bytes.
- `BufferCache::SynchronizeBufferFromImage` já oferece um caminho genérico image-to-buffer na GPU,
  incluindo imagens tiled, mips e tracking de range. Ele é uma base mais geral que o reconhecedor
  especial atual.
- `TextureCache::ProcessDownloadImages` chama `EndRendering` antes de saber se haverá cópia e
  suprime drains conservadores para qualquer entrada com authority em EOS, EOP, ReleaseMem ou
  `Other`. Essa supressão ampla não constitui prova semântica e não deve ser generalizada.
- `Liverpool::TrackDeferredGpuCompletion` não possui call site ativo no estado atual; não deve ser
  tratado como batching já implementado nem como solução comprovada.

Esse caminho faz quatro coisas importantes ao mesmo tempo:

1. preserva uma representação dos bytes produzidos pela GPU;
2. não grava imediatamente esses bytes na RAM guest;
3. não bloqueia imediatamente o GCP esperando o timeline tick;
4. mantém a publicação do label vinculada à conclusão física.

Essa combinação explica plausivelmente por que ele evita a corrupção vista ao simplesmente
remover `finish()`: a espera síncrona desaparece, mas a versão dos dados e a dependência lógica não
são descartadas. Isso ainda é uma **hipótese causal**, a ser confirmada pela instrumentação abaixo.

### 2.2 Evidência dos dumps existentes

Os dumps relevantes estão em
`D:\GAMES\EMULATION\EMULADORES\shadps4\user\log`:

- `shadps4-telemetry-1787533034499.csv.zst`
- `shadps4-telemetry-1787535992754.csv.zst`
- `shadps4-telemetry-1787536524709.csv.zst`
- `shadps4-telemetry-1787539932286.csv.zst`
- `shadps4-telemetry-1787540806119.csv.zst`
- `shadps4-telemetry-1787541418185.csv.zst`

O checkpoint funcional mais recente mostrou, nos 20 segundos finais:

- aproximadamente um readback legado por frame para o recurso 20;
- endereço guest `0x2B9A02C00`, 512 bytes, produtor `graphics_draw`, trigger EOS;
- 983 de 983 source-watches terminando como consumo GPU;
- nenhuma leitura semântica de CPU observada para essa versão;
- schedule-to-ready médio de aproximadamente 15,79 ms;
- host wait médio de aproximadamente 4,05 ms, com p95 próximo de um frame;
- memcpy de aproximadamente 1 microsegundo, portanto irrelevante frente à espera;
- cerca de sete `wait_progress` e onze submits Vulkan por frame;
- custo de CPU dentro do submit de aproximadamente 0,25 ms/frame;
- centenas de barreiras e quebras potenciais de rendering por frame.

Esses dados sustentam três conclusões:

1. há readbacks que não têm consumidor CPU e cujo custo principal é ordenação/espera, não cópia;
2. a cópia GPU-side é uma forma segura de preservar bytes enquanto o consumidor ainda é
   desconhecido;
3. o fast path pode ser ampliado, mas a ausência de leitura CPU em uma captura não é prova
   universal para descartar a versão.

### 2.3 O que ainda não foi provado

Ainda precisamos responder, com correlação completa:

- qual draw/dispatch escreveu cada candidato;
- qual pacote ou sequência de pacotes realmente concluiu esse produtor;
- quais ações de cache tornaram os dados visíveis ao próximo uso;
- se a cópia para shadow é necessária ou se a imagem original permanece válida até o consumidor;
- qual foi o primeiro consumidor semântico: image, buffer, CPU, overwrite, unmap ou nenhum;
- se o label foi observado por outro comando GPU, pela CPU, por IRQ ou por múltiplos consumidores;
- se o ganho vem principalmente de lazy materialization, da virtualização do wait, da remoção de
  `EndRendering`, da redução de submits ou de uma combinação;
- quais candidatos legados poderiam usar o mesmo contrato estrutural e por que os demais não
  poderiam.

As regras atuais por título, formato, 512 bytes e assinatura do EOS não respondem a essas
perguntas; elas apenas reconhecem uma ocorrência conhecida.

## 3. Contrato semântico derivado do SDK oficial

A fonte normativa para o guest é o SDK PS4 4.508.001 presente em
`D:\CODING\SDKs\PS4 SDK 4.50`. Os arquivos principais são:

- `sdk/target/include_common/gnm/constants.h`
- `sdk/target/include_common/gnm/drawcommandbuffer.h`
- `sdk/target/include_common/gnm/dispatchcommandbuffer.h`
- `sdk/target/include_common/gnmx/resourcebarrier.h`
- `sdk/target/src/gnmx/resourcebarrier.cpp`
- `sdk/target/samples/sample_code/graphics/api_gnm/synchronization-sample/*`
- `sdk/target/samples/sample_code/graphics/api_gnm/toolkit/toolkit.cpp`

### 3.1 Fatos do SDK que o avaliador precisa representar

| Operação guest | Garantia relevante | Limite da garantia |
| --- | --- | --- |
| `kEosCsDone` | Todo trabalho CS anterior ao ponto foi concluído | Não espera draws anteriores; deve seguir imediatamente o dispatch correspondente |
| `kEosPsDone` | Todo trabalho PS anterior ao ponto foi concluído | Não espera dispatches anteriores; deve seguir imediatamente o draw correspondente |
| `kEopFlushCbDbCaches` | CB/DB terminaram writes e flush de caches antes do evento | Não descreve sozinho o consumidor nem a autoridade do recurso |
| `kEopCbDbReadsDone` | CB/DB terminaram reads | Explicitamente não garante término dos writes |
| `kEopCsDone` | Alias numérico 0x28 usado para conclusão CS | O valor bruto 0x28 é ambíguo sem engine, sequência e uso do recurso |
| `kReleaseMemEventCsDone` | Trabalho CS anterior na fila compute foi concluído | O escopo de cache ainda depende dos campos do pacote |
| `waitOnAddress` | O command processor GPU espera a comparação na memória | O pacote não prova leitura dos dados do recurso pela CPU |
| `waitForGraphicsWrites` | Espera writes gráficos para slots/faixa e aplica ações de cache | A variante dispatch exige context roll e não espera compute writes |

O próprio `Gnmx::ResourceBarrier::write()` demonstra que não existe uma tradução correta baseada
apenas em opcode:

- RenderTarget sem DCC pode usar `waitForGraphicsWrites` restrito à faixa;
- RenderTarget com DCC usa label, EOP 0x28, wait e eventos de metadata;
- transições de RW texture/buffer usam EOP/ReleaseMem, cache action e wait;
- o mesmo valor 0x28 aparece com interpretações válidas diferentes;
- a ação muda conforme uso anterior, uso seguinte, tipo do recurso e metadata.

Consequentemente, o emulador precisa reconstruir o **contrato da sequência**, não reconhecer um
pacote isolado.

### 3.2 Sequências de referência do SDK

Os helpers oficiais fornecem sequências concretas para validar o decoder sem transformá-las em
novas assinaturas rígidas:

- `synchronizeComputeToGraphics`: zera um label, emite `kEosCsDone` imediatamente após o dispatch,
  espera o label e depois faz writeback/invalidate de L1/L2;
- `synchronizeComputeToCompute` no draw command buffer: EOS CS + wait + invalidate de L1;
- `synchronizeComputeToCompute` no dispatch command buffer: ReleaseMem CS + wait + invalidate de
  L1;
- `synchronizeRenderTargetGraphicsToCompute`: `waitForGraphicsWrites` com endereço, tamanho, slots
  CB e ações de cache específicas;
- `synchronizeGraphicsToCompute`: EOP que conclui rendering/flush, seguido de wait no label.

O decoder deve produzir scopes semanticamente equivalentes para essas sequências mesmo que o jogo
aloque outro label, use outro valor compatível, separe ações de cache em pacotes distintos ou use
um helper inline diferente. Os exemplos são oráculos de contrato, não fingerprints de pacote.

### 3.3 Cinco dimensões independentes

O novo modelo deve separar explicitamente:

1. **Conclusão:** quais engines/estágios e quais comandos anteriores terminaram.
2. **Visibilidade:** quais writes ficaram disponíveis para quais acessos/caches seguintes.
3. **Autoridade:** qual recurso, intervalo, alias e versão contém os bytes corretos.
4. **Representação:** imagem Vulkan, buffer Vulkan, shadow de download ou RAM guest.
5. **Observabilidade:** wait GPU, leitura CPU de label, leitura CPU dos dados, IRQ, unmap ou
   destruição.

Um label CPU-visible não torna automaticamente os dados associados CPU-consumed. Uma IRQ pode
exigir que a conclusão física e a publicação do evento sejam preservadas, sem exigir materializar
uma imagem. Analogamente, um `waitOnAddress` é uma dependência GPU mesmo quando o label reside em
memória endereçável pela CPU.

## 4. Arquitetura alvo

### 4.1 Fluxo único

```text
GPU write observado
    -> ReadbackCandidate
    -> pacote/sequência de sincronização decodificada em CompletionScope
    -> correlação causal candidate <-> scope
    -> avaliação de identidade, versão, range, aliases, snapshot e consumidor
    -> DataAction + SignalAction + SyncRequirementMask + EvidenceBits + RejectReasonMask
    -> commit transacional ou fallback legado
    -> acompanhamento até primeiro consumidor/overwrite/unmap/session end
```

`ScheduleComputeDownload` e `ScheduleRenderTargetDownload` deixam de decidir se algo “é o fast
path”. Eles apenas registram candidatos completos. EOS, EOP e ReleaseMem deixam de promover um
singleton; eles fecham escopos de conclusão e oferecem esses escopos ao avaliador.

### 4.2 Decisão dos dados e decisão do sinal são separadas

`DataAction`:

- `DirectGpuAuthority`: a versão continua na imagem/recurso GPU original; nenhuma cópia é emitida.
- `GpuShadow`: um comando GPU copia a imagem para um buffer Vulkan de download e preserva uma
  versão imutável sem gravar a RAM guest nem esperar o host naquele ponto.
- `LazyCpuMaterialization`: estado de uma authority/shadow quando surge demanda real de CPU.
- `LegacyRequired`: executa o readback atual, incluindo a espera exigida por seu contrato.

`SignalAction`:

- `PublishAfterPhysicalTick`: publica label/IRQ somente quando o timeline comprovar conclusão.
- `VirtualGpuWait`: satisfaz um wait guest correlacionado sem fazer o GCP aguardar o host.
- `ForceProgressSubmit`: submete trabalho pendente quando existe observador que não pode esperar o
  próximo submit natural.
- `ForceHostCompletion`: espera no host apenas quando CPU/IRQ/unmap exige conclusão imediata.
- `NoSignalAction`: não há label/evento relevante para esse candidato.

Combinações são possíveis. Por exemplo, uma IRQ pode exigir `PublishAfterPhysicalTick` enquanto os
dados permanecem em `DirectGpuAuthority`; um consumidor buffer pode exigir `GpuShadow` sem qualquer
label CPU-visible.

### 4.3 Requisito mínimo de sincronização

O resultado da classificação não deve ser um enum exclusivo do tipo “fast path ou legado”. Uma
transição pode precisar simultaneamente de ordem de execução, visibilidade de memória e mudança de
layout, sem precisar de host wait nem materialização. O avaliador produz uma máscara composável:

```text
SyncRequirementBits {
    ExecutionOrder,
    MemoryVisibility,
    ImageLayoutTransition,
    QueueOwnershipTransfer,
    HostSignalVisibility,
    CpuDataMaterialization,
    SnapshotPreservation,
    InterruptPublication
}
```

Máscara vazia significa que nenhum comando Vulkan ou ação de host adicional é necessário naquele
ponto. Os bits descrevem requisitos, não a implementação: `MemoryVisibility`, por exemplo, pode ser
satisfeito por ordem implícita válida, por uma barrier estreita no consumidor ou por uma transição
já necessária por outro recurso. Isso evita transformar cada evento guest em uma barrier isolada.

`SyncRequirementMask` complementa, sem substituir, `EvidenceBits` e `RejectReasonMask`:

- evidence prova o que sabemos;
- requirements dizem o efeito mínimo que precisa ser preservado;
- reasons dizem por que determinada `DataAction` ou `SignalAction` ainda não é segura;
- a ação escolhe a representação e o mecanismo mais baratos que satisfazem os requirements.

### 4.4 Hazards adiados e retenção do rendering scope

EOS, EOP, ReleaseMem, AcquireMem, SurfaceSync e EventWrite expressam partes do contrato guest. Eles
não devem emitir automaticamente uma barrier Vulkan ampla nem encerrar dynamic rendering. O
decoder registra a intenção de conclusão/visibilidade como um hazard pendente; o próximo consumidor
real resolve esse hazard usando:

```text
producer { resource/version/range, queue, stage, access, layout }
consumer { resource/version/range, queue, stage, access, layout }
    -> requirement mask
    -> dependência mínima producer -> consumer
```

A resolução pode concluir que basta ordem já implícita, que é necessária uma barrier estreita, que
há uma transição de layout/queue ou que faltam dados e o legado precisa ser mantido. Range, alias,
version ou consumidor desconhecido nunca autorizam omissão funcional; permanecem hazards pendentes
até um terminal seguro ou produzem fallback fail-closed.

O evento guest lógico não quebra sozinho o rendering scope. Toda chamada futura a
`Scheduler::EndRendering` deve receber um motivo e um `cause_id`:

```text
ScopeBreakReason {
    RequiredTransfer,
    RequiredMemoryDependency,
    RequiredLayoutTransition,
    RequiredHostVisibility,
    RequiredQueueTransfer,
    RequiredNonGraphicsCommand,
    AttachmentSetChange,
    Present,
    UnknownFallback
}
```

`UnknownFallback` preserva a correção e vira uma fila mensurável de investigação; não é tratado como
prova de que a quebra era obrigatória. Uma dependência que possa ser expressa legalmente dentro do
rendering atual, considerando as features habilitadas e as restrições de stage/access/layout, também
não deve forçar `EndRendering` por conveniência do call site.

Isso não autoriza mover barriers arbitrárias para dentro de dynamic rendering. Sem
`dynamicRenderingLocalRead`/features equivalentes, `vkCmdPipelineBarrier2` dentro de uma instância
iniciada por `vkCmdBeginRendering` é fortemente restrita e pode ser inválida; mesmo com a feature,
stages framebuffer-space, layouts, queue ownership e tipos de barrier continuam limitados. O maior
ganho inicial de retenção vem de provar que o evento lógico não exige comando naquele ponto. Quando
a dependência real for incompatível com o scope, `RequiredMemoryDependency` ou outro motivo preciso
justifica a quebra.

### 4.5 Registros mínimos

Os nomes finais podem mudar, mas os contratos precisam existir:

```text
ReadbackCandidateHot {
    candidate_id, image_id, image_uid, resource_epoch, alias_epoch,
    guest_begin, guest_end, subresource,
    producer_engine, producer_stage, producer_packet_seq,
    command_buffer_seq, capability_bits, state
}

CompletionScopeHot {
    scope_id, queue_id, engine, first_packet_seq, last_packet_seq,
    completed_stage_bits, completed_write_bits,
    visible_access_bits, cache_action_bits,
    guest_range_if_explicit, label_id, irq_bits, confidence
}

CandidateDecisionHot {
    candidate_id, scope_id, authority_id, producer_ticket,
    data_action, signal_action, sync_requirement_bits,
    evidence_bits, reason_mask, blocked_action_bits
}

PendingHazardHot {
    hazard_id, cause_id, resource_uid, resource_epoch, alias_epoch,
    guest_begin, guest_end, producer_queue_stage_access,
    required_bits, state
}

AuthorityVersionHot {
    authority_id, resource_uid, resource_epoch, alias_epoch,
    guest_begin, guest_end, representation_id,
    producer_ticket, state, consumer_count, generation
}

LogicalSignalHot {
    signal_id, label_addr, label_generation, value, size,
    compare/mask, producer_ticket, observation_bits, state
}

ScopeBreakHot {
    break_id, cause_id, command_buffer_seq, scope_id,
    reason, required_or_fallback, attachment_hash
}

CandidateDiagnosticCold {
    full image descriptor, raw packet controls, reason_mask,
    blocked_action_bits, evidence_bits, first_consumer,
    terminal_reason, timestamps
}

CausalCostRecordCold {
    cause_id, candidate_id, scope_id, packet_seq,
    decision, sync_requirement_bits, reason_mask, effect_bits,
    barrier_ids, scope_break_ids, copy_ids, submit_ids, wait_ids,
    gpu_interval_ids, attribution_confidence
}
```

O registro causal precisa representar relações muitos-para-muitos. Um submit pode servir vários
candidatos e uma decisão pode causar barrier, scope break, copy, submit e wait. Relatórios devem
atribuir custo exclusivo e compartilhado sem duplicar o mesmo intervalo GPU para cada causa.

### 4.6 Organização orientada a dados

O caminho quente deve usar lotes contíguos, IDs estáveis e generations. Não deve evoluir para um
grafo de `shared_ptr`, mutex e condition variable por candidato:

- candidatos do mesmo drain são percorridos linearmente;
- campos usados na classificação ficam juntos; descritores e strings de diagnóstico ficam frios;
- índices por image UID e páginas/ranges evitam varrer todas as authorities;
- um pool contíguo com handle `{index, generation}` evita ABA e pointer chasing;
- mutexes protegem pools/lotes, não cada registro individual;
- decisões, barriers e cópias são acumuladas antes do commit;
- sort/coalescing por `{resource_uid, epoch, guest_begin}` permite batching e boa localidade.

A primeira fase pode reutilizar estruturas existentes para reduzir risco, mas qualquer estrutura
nova deve respeitar esse destino. Não se deve consolidar o atual vetor de `shared_ptr` como API
permanente.

## 5. Provas, rejeições e política fail-closed

### 5.1 Evidence bits

Uma promoção só acontece com evidência positiva. Exemplos:

- produtor identificado por UID/epoch;
- scope posterior ao produtor na mesma fila;
- estágio do scope cobre o writer;
- ação de cache/visibilidade compatível com o próximo acesso;
- range/subresource coberto;
- imagem e alias generation ainda atuais;
- snapshot pode preservar a versão até todos os consumidores;
- label generation e wait correlacionados;
- consumidor CPU ausente até o ponto observado, ou materialização garantida sob demanda.

Os bits de evidência são registrados mesmo quando a decisão final é legado. Isso permite descobrir
classes quase elegíveis sem alterar comportamento.

### 5.2 RejectReasonMask

Uma máscara, e não um enum de motivo único, deve registrar todas as falhas relevantes.

Cada reason bit também carrega uma máscara das ações que ele bloqueia. Por exemplo,
`ImmediateCpuDataRead` bloqueia `DirectGpuAuthority`, mas pode permitir `GpuShadow` seguido de
materialização; `UnsupportedTiling` pode bloquear o shadow atual sem bloquear a permanência na
imagem; `PacketGapOrTraceLoss` bloqueia toda promoção funcional. O avaliador escolhe a ação mais
barata que permaneça comprovadamente segura, e não transforma qualquer reason bit em legado por
definição.

**Ordenação/PM4**

- `NoCompletionScope`
- `ProducerUnknown`
- `ProducerAfterScope`
- `StageNotCovered`
- `QueueOrderUnknown`
- `CrossQueueDependencyMissing`
- `PacketGapOrTraceLoss`
- `AmbiguousEventSemantics`
- `CacheVisibilityInsufficient`
- `RangeNotCoveredByScope`

**Agendamento/configuração**

- `ReadbackDisabledByConfiguration`
- `GuestAddressUnavailable`
- `ResourceNotGpuModified`
- `SupersededBeforeEvaluation`

**Identidade/linhagem**

- `ImageFreedOrReused`
- `ResourceEpochChanged`
- `AliasEpochChanged`
- `AliasWriterAmbiguous`
- `PartialOverlapAmbiguous`
- `TopologyChanged`
- `MultipleVersionsRequired`

**Capacidade de representação**

- `ImageNotSafeToDownload`
- `UnsupportedTiling`
- `UnsupportedFormatOrAspect`
- `UnsupportedMipLayerRegion`
- `CopyRegionNotRepresentable`
- `SnapshotAllocationFailed`
- `PinOrLifetimeUnavailable`
- `StagingPressureLimit`

**Consumidor/observabilidade**

- `ImmediateCpuDataRead`
- `CpuPartialWriteNeedsPreservation`
- `UnknownConsumerWithoutDurableSnapshot`
- `LabelReadByCpuBeforeNaturalSubmit`
- `IrqRequiresCompletion`
- `MultipleSignalConsumers`
- `UnsupportedWaitComparison`
- `LabelGenerationMismatch`
- `LabelAddressAliased`

**Ciclo de vida**

- `UnmapBeforeCompletion`
- `RemapOrAbaRisk`
- `ShutdownInProgress`
- `DeviceLost`
- `AuthorityPressureEviction`
- `InternalValidationFailure`

O schema atual (`ProducerNotCompute`, `ResourceNotStorage`, `SizeNot3072`, etc.) descreve
experimentos antigos. Ele deve ser substituído por essa taxonomia estrutural. Tamanho, formato e
título continuam sendo campos de agrupamento, nunca o motivo semântico principal.

### 5.3 Regra para consumidor desconhecido

Consumidor desconhecido não autoriza `DirectGpuAuthority`. Há duas respostas seguras:

1. usar `GpuShadow` se a cópia preservar integralmente a versão, o pin sobreviver até o último
   consumidor e qualquer leitura CPU posterior puder materializá-la;
2. usar `LegacyRequired` se nenhuma representação durável puder ser garantida.

Essa é a primeira ampliação de baixo risco: um shadow conserva os mesmos bytes que o readback
legado copiaria, mas adia a espera e a escrita em RAM. A authority direta vem depois, quando a
linhagem e o consumidor GPU forem comprovados.

### 5.4 Classificação de sincronização evitável

Para diagnóstico, cada efeito observado ou contrafactual recebe uma destas classes:

- `ProvenRequired`: existe consumidor/observador identificado que exige o efeito;
- `ProvenEliminable`: o ledger prova que uma ação mais barata preserva a mesma observação;
- `ConservativeFallback`: a execução atual usa o efeito porque falta prova para removê-lo;
- `UnknownDueToTraceGap`: a captura perdeu a informação necessária para classificar.

Essa classificação é telemetria, não um atalho funcional. Não existe categoria “provavelmente
obrigatório” capaz de autorizar mudança de comportamento. Somente prova positiva promove uma ação;
`ConservativeFallback` e `UnknownDueToTraceGap` continuam executando o legado. A métrica útil é a
conversão de fallback conservador em requisito provado ou efeito eliminável, e não a mera redução
de contadores.

## 6. Instrumentação necessária antes de generalizar

### 6.1 Fase observe-only

A primeira implementação não altera nenhuma decisão funcional. O fast path conhecido continua
funcionando exatamente como hoje; todos os demais candidatos seguem pelo legado. Em paralelo, o
avaliador calcula a decisão contrafactual e registra por que promoveria ou rejeitaria.

Eventos mínimos:

1. `candidate_schedule`
   - candidate ID, UID, epoch, alias epoch, endereço/range;
   - descriptor estrutural: tiled, format, aspect, dimensões, pitch, mips, layers;
   - writer/engine/stage, producer seq, packet seq, command buffer seq;
   - capability bits para direct/shadow/materialization.
2. `completion_scope`
   - EOS/EOP/ReleaseMem/SurfaceSync/AcquireMem participantes;
   - engine, queue, estágio concluído, writes cobertos, cache actions;
   - first/last packet, label, IRQ e confidence;
   - digest da janela PM4 causal, sem despejar o command buffer inteiro.
3. `candidate_decision`
   - `DataAction` e `SignalAction` contrafactuais;
   - sync requirement mask, evidence bits e reason mask completos;
   - classe de evitabilidade para cada efeito que a execução real produzir;
   - IDs do candidate, scope, ticket e sinal correlacionados.
4. `candidate_representation`
   - imagem original, shadow, buffer alias ou RAM;
   - copy bytes, allocation/pin, timeline tick e submit seq.
5. `candidate_consumer`
   - primeiro consumidor semântico e todos os consumidores posteriores relevantes;
   - CPU/GPU, image/buffer, range, layout/access, producer packet e destination UID;
   - se consumiu a mesma versão ou exigiu materialização.
6. `candidate_terminal`
   - materializado, consumido GPU, sobrescrito, destruído, unmapped, superseded ou session end;
   - tempo de vida, bytes preservados e waits/cópias evitáveis contrafactuais.
7. `logical_signal`
   - criação, geração, publicação, wait correlacionado, leitura CPU, IRQ e retirement;
   - se o produtor já estava submitted/completed quando cada observação ocorreu.
8. `hazard_resolution`
   - produtor e consumidor por UID/epoch/alias/range/version;
   - stage/access/layout/queue de ambos e requirement mask resultante;
   - barrier emitida, dependência implícita reutilizada ou motivo do fallback.
9. `scope_break`
   - rendering scope, attachment hash, `ScopeBreakReason`, `cause_id` e command buffer;
   - se a quebra era provada, conservadora ou desconhecida por trace gap.
10. `causal_effect`
    - liga a decisão original às barriers, copies, scope breaks, flushes, submits, waits e intervalos
      GPU que ela provocou ou compartilhou.

### 6.2 Correlação obrigatória

Todo evento precisa carregar IDs relacionais, não depender de proximidade temporal:

- `candidate_id`
- `resource_uid/resource_epoch/alias_epoch`
- `producer_seq/packet_seq`
- `scope_id`
- `command_buffer_seq/submit_seq/timeline_tick`
- `authority_id/representation_id`
- `signal_id/label_generation/wait_seq`
- `consumer_id`
- `hazard_id/barrier_id/scope_break_id/cause_id`
- `gpu_interval_id/query_frame_id`

Uma captura com ring overwrite, packet gap ou perda de uma dessas chaves não pode classificar o
candidato como seguro. Ela deve produzir `PacketGapOrTraceLoss`.

### 6.3 Custo da instrumentação

- registros fixed-size em ring buffers pré-alocados;
- nenhuma formatação, compressão ou I/O no GCP;
- writer separado para CSV/Zstandard;
- dados frios e descritores completos emitidos uma vez por candidate/descriptor hash;
- eventos de alta frequência agregados ou amostrados, preservando todos os eventos de decisão;
- timestamp Vulkan amostrado e lido em frames posteriores, nunca com wait para obter query;
- contador de registros vistos, retidos, sobrescritos e filtrados por stream.

O build normal deve compilar esse caminho para no-op pela flag existente. O build instrumentado
não serve como referência absoluta de FPS; ele serve para comparar causalidade entre builds
instrumentadas equivalentes.

### 6.4 Cadeia causal de custo

Contar barriers ou readbacks isoladamente não explica o custo. Cada `cause_id` deve acompanhar a
cadeia inteira:

```text
candidate/scope/packet
    -> decisão + requirements + reasons
    -> hazard/barrier
    -> scope break
    -> copy/resolve/tile
    -> flush/submit
    -> host wait ou gap de queue
    -> intervalos GPU afetados
```

Uma causa pode produzir vários efeitos e um efeito pode servir várias causas. O agregador deve:

- registrar `exclusive`, `shared` ou `unknown` para a atribuição;
- contar uma barrier/submit/espera compartilhada uma única vez no total global;
- distribuir ou exibir separadamente custo compartilhado, nunca replicá-lo integralmente por reason;
- impedir soma ingênua de intervalos GPU sobrepostos;
- distinguir tempo inclusivo do intervalo, tempo exclusivo conhecido e tempo não atribuído;
- carregar confidence e trace health até o relatório final.

O relatório principal passa a ordenar reason masks não apenas por frequência, mas por efeitos
resultantes: barriers, rendering breaks, copies, submits, waits, bytes e tempo GPU exclusivo/
compartilhado. Uma correlação temporal sem IDs é insuficiente para declarar causalidade.

### 6.5 Profiler GPU em camadas

O uso próximo de 100% da RTX 3060 não prova sozinho que o guest “precisa” daquele custo. Um emulador
pode amplificar trabalho, gerar shaders host menos eficientes, criar passes/copies/resolves extras,
aumentar overdraw, usar resolução maior, serializar dependências ou fragmentar command buffers. A
comparação de FLOPS com a GPU do PS4, isoladamente, também não identifica qual desses fatores está
limitando o frame. A medição precisa separar tempo, volume de trabalho e utilização interna.

Na máquina de referência, `vulkaninfo` registrou em 2026-08-24:

| Capacidade | RTX 3060 local | Uso no plano |
| --- | --- | --- |
| timestamps na graphics queue | 64 bits, período de 1 ns | profiler contínuo amostrado |
| `pipelineStatisticsQuery` | disponível | volume de trabalho por bloco amostrado |
| `VK_EXT_calibrated_timestamps` | disponível | correlação CPU/GPU independente do Tracy |
| `VK_KHR_pipeline_executable_properties` | disponível | diagnóstico estático de pipelines |
| `VK_KHR_dynamic_rendering_local_read` | disponível, ainda não habilitada no backend | A/B restrito para dependências framebuffer-local |
| `VK_KHR_performance_query` | não exposta pelo driver | não depender dela para counters de hardware |

Esses valores são baseline, não contrato: a implementação consulta features/properties no startup e
desativa cada camada quando indisponível.

#### 6.5.1 Camada 1 — timestamps contínuos e baratos

Usar `vkCmdWriteTimestamp2` com um ring de query pools dimensionado pelos frames em voo:

- amostrar 1/N frames, com N configurável e budget máximo de queries;
- resetar/reutilizar slots somente depois do retirement do frame correspondente;
- ler resultados alguns frames depois com availability bit, sem `WAIT_BIT` e sem host wait;
- descartar amostras ainda indisponíveis e contabilizar a perda;
- calibrar relógios CPU/GPU com `VK_EXT_calibrated_timestamps` quando disponível, sem condicionar o
  recurso a `TRACY_GPU_ENABLED`;
- usar registros fixed-size e resolver nomes/hashes fora do GCP.

Brackets mínimos:

- frame guest e command buffer;
- cada dynamic rendering scope;
- blocos de draws agrupados por pass/attachment set/pipeline hash;
- blocos de dispatches por pipeline hash;
- copy, tile, detile, resolve e clear auxiliares;
- dependências/barriers agrupadas por `cause_id` e requirement mask;
- fim/início de command buffers consecutivos e present.

Não inserir queries em cada draw na coleta normal. O bloco deve se subdividir adaptativamente apenas
quando ultrapassar um budget de tempo ou comandos, permitindo localizar o pipeline caro sem tornar
a telemetria a causa do gargalo.

Um timestamp antes/depois de uma barrier **não mede uma duração intrínseca da barrier**. O intervalo
pode conter trabalho produtor anterior que a dependência precisa aguardar e outras sobreposições. O
schema deve chamá-lo `dependency_delay_interval`, tratá-lo como janela de serialização e confirmar a
causa por correlação/A-B. Intervalos GPU podem se sobrepor; não podem ser somados como categorias
disjuntas sem cálculo de união/exclusividade.

#### 6.5.2 Camada 2 — pipeline statistics e amplificação host

Em uma amostra ainda menor, abrir queries de pipeline statistics por rendering scope ou bloco de
pipeline para obter, quando suportado, primitives/vertices, shader invocations, clipping e compute
invocations. Essas queries têm overhead e nuances de implementação; servem para comparação relativa
entre A/B equivalentes, não para telemetria em todos os frames.

Combinar counters guest já decodificados com counters host:

- host draws por guest draw;
- host dispatches por guest dispatch;
- rendering scopes/passes host por pass guest;
- barriers Vulkan por evento de sincronização guest;
- fragment shader invocations por pixel final e por draw;
- bytes de copy/resolve/tile por bytes realmente modificados no guest;
- primitives e invocations por hash de pipeline e attachment set.

Essas razões detectam amplificação causada pela tradução. `fragmentShaderInvocations/output_pixels`
é um indicador relativo de overdraw e quads auxiliares, não uma igualdade exata entre invocação e
pixel devido às regras de rasterização da implementação.

#### 6.5.3 Camada 3 — diagnóstico estático de pipeline

Usar `VK_KHR_pipeline_executable_properties` somente em uma build diagnóstica curta para associar
pipeline/shader hash a executáveis e estatísticas expostas pelo driver, como instruções, registers ou
spills quando disponíveis. Os flags de captura podem elevar o custo de criação e interferir no
pipeline cache; portanto:

- nunca habilitar essa captura na build normal ou no dump de frame time principal;
- aquecer/compilar os pipelines antes da cena medida;
- tratar os dados como propriedades estáticas do executável, não como tempo dinâmico;
- usar os hashes para cruzar os pipelines estáticos com os blocos caros das camadas 1 e 2.

#### 6.5.4 Camada 4 — validação externa curta

Para uma captura representativa e curta, usar NVIDIA Nsight Graphics GPU Trace, disponível para a
RTX 3060, a fim de verificar unidade limitante, throughput, underutilization e sincronização entre
queues. Inserir `VK_EXT_debug_utils` labels com frame, pass, attachment hash, pipeline hash,
`candidate_id`, `cause_id` e fallback reason para ligar a captura externa à telemetria do emulador.

O driver local não expõe `VK_KHR_performance_query`; logo, occupancy, bandwidth e utilização de
unidades devem vir dessa ferramenta externa, não de uma falsa aproximação no CSV. RenderDoc continua
útil para conferir conteúdo, layouts e estrutura de passes, mas não substitui a medição temporal do
GPU Trace.

### 6.6 Matriz para diagnosticar a cena GPU-limited

Executar a mesma rota/save/câmera, driver e cache de shaders já aquecido. Capturar a janela relevante
em 0,5x, 1x e 2x da resolução base, primeiro em build normal para FPS/frame pacing e depois em build
instrumentada equivalente para causalidade. O overhead da instrumentação precisa ser medido em uma
cena-controle e subtraído apenas como contexto, nunca usado para “corrigir” artificialmente tempos.

| Resultado observado | Hipótese principal | Confirmação seguinte |
| --- | --- | --- |
| tempo cresce aproximadamente com pixels | fragment shading, overdraw ou attachment bandwidth | invocations/pixel, top pipelines, Nsight throughput |
| tempo quase constante com resolução | compute, geometry, copies fixas, barriers ou fragmentação | dispatch/copy time, dependency intervals, scope/submit count |
| poucos hashes dominam o frame | shader host ineficiente ou pass legítimo caro | executable stats/disassembly e A/B do pipeline |
| host/guest work ratio alto | amplificação da tradução | localizar criação de passes/draws/copies extras |
| dependency intervals dominam | oversynchronization ou produtor realmente longo | requirements, próximo consumidor e A/B de barrier estreita |
| muitos scopes/command buffers pequenos | `EndRendering`/flush conservador | `ScopeBreakReason` e cadeia causal do submit |
| copies/tile/resolve dominam | representação/layout inadequado ou readback residual | bytes, origem, destino, consumidor e possibilidade de authority |

O diagnóstico termina com um ranking por pass/pipeline hash e `cause_id`, contendo tempo inclusivo,
exclusivo e compartilhado, volume de trabalho, razão guest/host e confidence. “GPU 99%” sem esse
ranking não é conclusão causal.

### 6.7 Perguntas que o primeiro dump novo deve responder

- Quantos downloads legados foram considerados?
- Quantos teriam `DirectGpuAuthority`, `GpuShadow` ou `LegacyRequired`?
- Para cada legado, quais reason bits e requirements impediram promoção?
- Qual foi o primeiro consumidor de cada classe estrutural?
- Quantos candidatos morreram sem qualquer consumidor CPU?
- Quantos shadows teriam sobrevivido a uma troca de epoch/alias?
- Quantos `EndRendering`, barriers, copies, waits e submits foram provados, conservadores ou
  elimináveis por decisão?
- Quais `ScopeBreakReason` dominam e quantos são `UnknownFallback`?
- Quantos waits virtuais ainda causariam `WaitProgress`, e qual observador exigiu esse submit?
- Quais scopes têm correlação completa com o producer e quais dependem de heurística?
- Quais efeitos foram compartilhados e qual seria a economia sem dupla contagem?
- Quanto tempo GPU pertence a rendering/draw/dispatch, copy/tile/resolve, dependency delay e gaps?
- Quais pass/pipeline hashes dominam os frames lentos e qual amplificação guest->host apresentam?
- O custo escala com resolução, invocations, bytes transferidos, barriers ou fragmentação?

## 7. Investigação causal do fast path atual

Antes de ampliar a classe aceita, a implementação conhecida deve ser decomposta em hipóteses.

### Hipótese A — o ganho vem da remoção da espera CPU/GCP

Evidência esperada: o shadow é gravado no command buffer, o GCP continua, e nenhuma demanda CPU
surge antes do próximo submit natural. O legado mostra host wait no mesmo ponto.

### Hipótese B — o shadow é o que preserva a correção

Evidência esperada: a imagem original muda de epoch, é reusada ou sofre alias antes do primeiro
consumidor; o shadow continua contendo os bytes da versão anterior. Se a imagem original permanece
estável até o consumidor, direct authority pode ser suficiente para essa classe.

### Hipótese C — o wait guest é somente GPU-to-GPU

Evidência esperada: o label é consumido por `WAIT_REG_MEM`, não pela CPU, e a dependência seguinte
é um acesso GPU correlacionado. A publicação física do label pode ocorrer depois sem alterar uma
observação intermediária.

### Hipótese D — o ganho inclui evitar fragmentação

Evidência esperada: o fast path reduz `EndRendering`, command buffer rollover e `WaitProgress`, não
apenas o tempo de `MasterSemaphore::Wait`.

### Método de isolamento

Depois do observe-only, e somente para a classe atual já funcional, criar modos diagnósticos A/B
que alterem uma dimensão por vez:

1. legado completo;
2. fast path funcional atual;
3. shadow/lazy materialization mantidos, mas wait virtual desativado;
4. wait virtual mantido, mas publicação física forçada no próximo submit já necessário;
5. shadow mantido, com materialização assíncrona antecipada após completion, sem bloquear o GCP.

Cada modo precisa preservar os mesmos bytes e labels; nenhum experimento pode equivaler a remover
`finish()` sem representação substituta. A comparação deve usar a mesma cena, build instrumentada,
driver e janela temporal.

## 8. Fases de implementação

### Fase 0 — Congelar a referência e estabelecer o profiler

**Objetivo:** impedir que a investigação perca o estado funcional.

- criar checkpoint local antes da primeira mudança de código;
- registrar branch/commit, CMake flags, driver, config do jogo e hashes dos executáveis;
- manter um switch experimental que retorna imediatamente ao comportamento conhecido;
- guardar um save e uma rota reproduzível para loading, shock wave e cena GPU-heavy;
- validar uma run normal e uma instrumentada no checkpoint;
- registrar as capabilities Vulkan da máquina e implementar primeiro a camada 1 do profiler sem
  alterar decisões de readback, barrier, scope ou submit;
- medir o overhead por taxa de amostragem e definir um budget que não introduza host waits;
- gerar uma baseline da cena GPU-heavy em 0,5x/1x/2x com shader cache aquecido.

**Gate:** loading, correção visual e performance do checkpoint reproduzidos; timestamps correlacionam
CPU/GPU sem wait, e o overhead/perda de queries estão quantificados.

### Fase 1 — Candidatos e scopes em observe-only

**Objetivo:** observar todos os caminhos sem alterar a execução.

- como pré-requisito isolado, corrigir a coerência de `PageState::num_read_watchers`: hoje
  `ApplyPageDelta` altera o campo sob `read_watch_mutex`, enquanto `HasReadWatcher` o lê sem o mesmo
  lock. Nenhuma decisão de authority pode depender de um read-watch com data race;
- incorporar os metadados de `PendingFastpathCandidate` em cada `PendingImageDownload`;
- parar de usar o singleton como fonte de telemetria;
- registrar toda tentativa de schedule antes dos filtros atuais e criar um candidato para toda
  solicitação conservadora válida, inclusive as que continuarão legadas;
- decodificar EOS, EOP, ReleaseMem, SurfaceSync/AcquireMem e waits em `CompletionScope`;
- registrar reason mask, evidence bits, sync requirements e decisão contrafactual;
- registrar hazards pendentes, resoluções contrafactuais e `ScopeBreakReason` sem mudar a execução;
- ligar cada candidato aos efeitos reais por `cause_id`, inclusive efeitos compartilhados;
- acompanhar primeiro consumidor e terminal state;
- classificar antes de `scheduler.EndRendering()`; no observe-only, executar o legado depois;
- integrar timestamps e pipeline statistics amostrados aos IDs causais.

**Não fazer:** fora da correção isolada do watcher, mudar promoção, authority, wait, label, barrier
ou materialização.

**Gate:** dump sem lacunas relacionais relevantes, overhead compreendido e comportamento idêntico.

### Fase 2 — Explicar o caso funcional

**Objetivo:** transformar o fast path atual de assinatura conhecida em uma classe causal.

- reconstruir a janela PM4 entre producer, completion scope, label, wait e primeiro consumidor;
- comparar essa sequência com as garantias do SDK;
- executar os modos A/B mínimos se os eventos observacionais não isolarem o ganho;
- documentar quais evidence bits são necessários e quais campos atuais eram acidentais;
- produzir uma tabela `classe estrutural -> ação segura -> prova -> risco residual`;
- explicar a cena GPU-heavy com scaling de resolução, top pass/pipeline hashes, amplificação
  guest->host, dependency intervals e custo de copy/tile/resolve.

**Gate:** ser possível explicar por que os bytes e labels permanecem corretos e qual operação
remove o custo; e localizar o trabalho que domina os frames GPU-limited. “Funciona em CUSA01715” ou
“a GPU está em 99%” não satisfaz o gate.

### Fase 3 — Generalizar primeiro o GpuShadow

**Objetivo:** ampliar o caminho de menor risco sem aceitar direct authority prematuramente.

- remover filtros por título, tamanho, formato e valor do label da decisão estrutural;
- inicialmente aceitar somente imagens cuja copy region e materialização já sejam representáveis;
- validar todos os candidatos de um lote antes de emitir a primeira cópia;
- reservar staging/pins e construir barriers antes de remover pendências;
- emitir image-to-buffer copies agrupadas e publicar authorities em um commit único;
- manter cada versão viva até CPU read, GPU consumer, overwrite seguro ou terminal state;
- em allocation/pin/identity failure, deixar o lote intacto e executar o legado;
- manter um gate de rollout por configuração/título se necessário, mas sem regras de correção por
  título. O gate limita exposição; o avaliador continua genérico.

**Gate:** zero corrupção/loading hang, equivalência de bytes em materializações amostradas e queda
reproduzível dos waits legados nas classes promovidas.

### Fase 4 — Authority por intervalo, versão e consumidor

**Objetivo:** permitir consumo GPU sem passar por RAM e habilitar direct authority quando provado.

- identificar authority por UID + resource epoch + alias epoch + intervalo/subresource;
- dividir overlaps parciais em intervalos; nunca satisfazer uma faixa inteira com prova parcial;
- preservar versões antigas enquanto houver consumidor anterior à nova write;
- para image consumer, manter a imagem e aplicar somente a dependência Vulkan necessária;
- para buffer consumer, usar `BufferCache::SynchronizeBufferFromImage` ou o shadow já existente;
- para CPU read, materializar somente o overlap solicitado;
- para CPU full overwrite, aposentar bytes cobertos sem readback;
- para CPU partial write, preservar/materializar somente bytes não sobrescritos;
- usar shadow para consumidor ainda desconhecido; usar direct authority apenas quando a imagem e a
  versão sobreviverem até o consumidor comprovado.

**Gate:** todos os primeiros consumidores classificados, sem stale guest upload e sem perda de
versão em alias/rebind/ABA.

### Fase 5 — Resolver hazards no consumidor e reter rendering scopes

**Objetivo:** eliminar sincronização falsa antes de tentar agrupar seus efeitos residuais.

- converter o contrato guest decodificado em `PendingHazard`, sem emitir imediatamente uma barrier
  ampla no packet handler;
- resolver o hazard quando o próximo consumidor do mesmo UID/epoch/alias/range/version for conhecido;
- derivar stage/access/layout/queue mínimos do producer e consumer reais;
- reutilizar dependência/ordem já existente quando ela satisfizer o requirement, sem duplicar barrier;
- manter dynamic rendering aberto quando o evento guest for apenas lógico ou a dependência puder
  ser expressa legalmente dentro do scope com as features efetivamente habilitadas;
- exigir `ScopeBreakReason + cause_id` para toda quebra e centralizar o ponto de decisão;
- encerrar rendering somente para transfer/layout/queue/host visibility/comando incompatível,
  attachment change, present ou fallback desconhecido real;
- promover funcionalmente apenas `ProvenEliminable`; `ConservativeFallback` e trace gaps mantêm o
  comportamento legado;
- comparar A/B, uma classe por vez, começando pelos maiores custos exclusivos comprovados.

**Gate:** nenhuma dependência ausente em validation/captura; redução explicável de barriers,
`EndRendering`, command buffer rollovers e tempo GPU/GCP, sem mudança de bytes/layout/ordem guest.

### Fase 6 — Generalizar sinais e waits

**Objetivo:** desacoplar conclusão guest de host wait sem confundir label com data readback.

- associar cada logical signal a um `CompletionScope` e a um timeline ticket;
- suportar múltiplos waits/consumidores com label generation explícita;
- virtualizar apenas comparadores/máscaras cujo resultado seja exatamente reproduzível;
- preservar publicação física do label e IRQ na ordem correta;
- uma IRQ força completion/publication quando necessário, mas não força materialização dos dados;
- leitura CPU do label pode forçar progress submit/completion do sinal, sem materializar recursos
  não lidos;
- remover a suposição de uma active virtual fence por endereço quando gerações coexistirem;
- tratar cross-queue com ticket e dependência explícitos; ausência de prova volta ao legado;
- adiar `WaitProgress` quando producer e consumer podem permanecer ordenados no mesmo command
  buffer e não existe observador intermediário.

**Gate:** labels, waits e IRQs idênticos ao legado; nenhum stale callback; redução de
`WaitProgress` sem aumentar latência de observações CPU.

### Fase 7 — Batching residual de readbacks, barriers e submits

**Objetivo:** reduzir o custo dos efeitos que continuam comprovadamente necessários depois que
hazards falsos, scope breaks e flushes artificiais tiverem sido eliminados.

- agrupar candidatos por command buffer, completion scope, resource epoch e compatibilidade de
  copy/barrier;
- executar um único `EndRendering` apenas se o lote realmente emitir copy ou barrier incompatível
  com rendering;
- coalescer ranges contíguos sem misturar epochs ou aliases;
- reservar uma allocation maior e emitir múltiplas regiões quando isso reduzir overhead;
- compartilhar um timeline tick entre logical signals somente quando não houver observação entre
  eles;
- manter trabalho no command buffer atual até um trigger real: present, pressão, dependência
  cross-queue, CPU label read, IRQ ou host data read;
- registrar por que cada submit ocorreu e quais sinais/readbacks ele tornou observáveis;
- estreitar stage/access/range de barriers com base no ledger, nunca por remoção heurística.

Submit batching é limpeza posterior, não a fonte primária de ganho. Se um submit existe apenas
porque uma barrier/download conservador encerrou o command buffer, a Fase 5 deve eliminar a causa em
vez de tornar o submit artificial mais barato.

**Gate:** menos rendering breaks/submits/barriers no build instrumentado e melhora de p95/p99 no
build normal, sem aumentar GPU time útil nem criar starvation.

### Fase 8 — Vulkan moderno e diagnóstico externo, somente após o modelo causal

O código já usa timeline semaphore e `pipelineBarrier2`; portanto:

- `VK_KHR_timeline_semaphore` continua sendo o ticket físico principal;
- synchronization2 permite scopes mais precisos, mas não descobre hazards automaticamente;
- migrar para `vkQueueSubmit2` pode tornar stage scopes mais explícitos, mas não reduz submits nem
  garante ganho por si só;
- timestamp e pipeline-statistics queries são a base do profiling interno;
- `VK_EXT_calibrated_timestamps` deve correlacionar CPU/GPU independentemente do Tracy;
- `VK_KHR_pipeline_executable_properties` fica restrita a builds diagnósticas por pipeline;
- `VK_KHR_dynamic_rendering_local_read` merece A/B para dependências framebuffer-local na RTX 3060,
  mas não autoriza transfer, queue ownership, layout transition arbitrária ou stages não gráficos;
  habilitá-la só é útil depois que o ledger separar essas classes;
- como o driver de referência não expõe `VK_KHR_performance_query`, counters de occupancy,
  bandwidth e unidades limitantes devem ser validados em captura curta do Nsight GPU Trace;
- `VK_KHR_maintenance9` pode reduzir ownership transfers de buffers/linear images em arquiteturas
  multi-queue, mas não elimina readback nem espera de consumidor;
- `VK_EXT_host_image_copy` é inadequado ao caminho GPU-only e pode alterar as características de
  acesso/compressão da imagem; só deve ser testado para materialização CPU comprovada e após query
  de performance do formato/usage;
- internally synchronized queues só atacam sincronização externa da queue. O mutex/submit CPU
  medido é pequeno, então o ROI esperado é baixo;
- nenhuma extensão substitui o ledger de authority, versão, visibilidade e observabilidade.

Cada extensão deve ser um A/B isolado, condicionada a feature/property do driver e com fallback.

### Fase 9 — Consolidação orientada a dados

**Objetivo:** remover a dívida experimental sem alterar semântica.

- substituir singleton, vectors de `shared_ptr` e mutex por entrada por pools/lotes compactos;
- separar metadados frios de telemetria dos campos quentes;
- medir lookup por range, cache misses e contenção antes de escolher índice definitivo;
- remover enums/reasons obsoletos de experimentos 3K/compute-only;
- eliminar APIs duplicadas entre fast path especial e caminho genérico;
- reduzir logs do GCP; manter apenas counters/eventos fixed-size atrás da flag;
- remover title checks da lógica de correção; manter apenas controles explícitos de rollout;
- revisar shutdown, device lost, unmap e pressure eviction.

**Gate:** mesmo comportamento da fase anterior, menor complexidade e nenhum custo mensurável no
build normal com o recurso desativado.

## 9. Invariantes obrigatórios

1. Endereço e tamanho nunca identificam sozinhos uma authority.
2. `ImageId` reciclável nunca substitui `image_uid + epoch + generation`.
3. Nenhuma pendência é removida antes de validar e reservar todo o lote.
4. Nenhuma falha parcial publica metade das authorities ou metade dos sinais.
5. Uma nova write fecha a versão anterior para novos consumidores, mas não destrói a versão ainda
   exigida por consumidores antigos.
6. Um shadow permanece pinned até conclusão física e último consumidor.
7. Materialização espera somente o ticket que produz a versão solicitada, não o tick global mais
   novo.
8. CPU lê somente bytes materializados da versão correta.
9. GPU consumer nunca lê RAM guest stale quando há imagem/shadow autoritativo.
10. IRQ e label visibility não implicam data materialization.
11. Wait virtual só é consumido uma vez por observação lógica, com suporte explícito a múltiplos
    consumidores quando o contrato exigir.
12. Callback tardio verifica label generation, authority generation e device state.
13. Trace incompleto nunca promove um candidato.
14. Fallback legado permanece disponível em qualquer fase.
15. A consulta de read-watch é sincronizada com arm/disarm; um bit de página sem owner/generation
    não serve como prova de consumidor de uma authority específica.
16. Um pacote guest de sincronização cria/fecha escopo semântico, mas não implica por si só barrier,
    `EndRendering`, flush, submit ou host wait.
17. Todo hazard pendente termina resolvido contra um consumidor/version/range identificado ou em
    fallback explícito; não pode desaparecer ao trocar command buffer.
18. Toda quebra de rendering, barrier, copy, submit e wait originada por esse mecanismo carrega
    `cause_id` e classificação de necessidade.
19. Intervalos GPU sobrepostos ou efeitos compartilhados nunca são somados repetidamente no total.

## 10. Casos de borda a validar

| Caso | Comportamento obrigatório |
| --- | --- |
| CPU e GPU consomem a mesma versão | GPU usa authority; CPU materializa o subrange e espera o mesmo ticket |
| CPU full overwrite | aposenta somente a faixa coberta, sem readback desnecessário |
| CPU partial write | preserva a parte não sobrescrita e cria nova versão para a escrita |
| Overlap parcial de duas authorities | particiona ou rejeita; nunca escolhe arbitrariamente |
| Alias/rebind | segue backing/writer/download por alias epoch |
| Múltiplos mips/layers | representa cada subresource/copy region corretamente |
| Tiled/depth/stencil/metadata | usa tile/resolve compatível ou rejeita com motivo preciso |
| Allocation failure | nenhum efeito parcial; fallback legado intacto |
| Ring wrap/reclaim | pin impede reciclagem antes do último ticket/consumer |
| Label reutilizado | generation impede callback ABA |
| Equal/GE/masked/non-equal waits | virtualiza somente a função implementada exatamente |
| Múltiplos waits no mesmo label | mantém contagem/consumidores, não um boolean singleton |
| IRQ sem CPU data read | conclui e publica o sinal; dados continuam GPU-side |
| CPU lê label durante command buffer aberto | força progress do sinal, não readback dos recursos |
| Cross-queue graphics/compute | dependência explícita por queue/ticket; caso contrário legado |
| Mesmo attachment volta a ser usado | mantém scope/layout quando válido; não quebra por evento lógico intermediário |
| Próximo consumidor ainda desconhecido | conserva hazard/authority ou usa shadow; não omite dependência por ausência momentânea |
| Barrier atende vários recursos | uma emissão, múltiplas causas compartilhadas, sem duplicar custo |
| Transfer/dispatch dentro de rendering | quebra com `RequiredTransfer`/`RequiredNonGraphicsCommand` e restaura estado corretamente |
| Unmap/remap | completa ou invalida somente authorities sobrepostas e suas gerações |
| Shutdown/device lost | callbacks não tocam handles/RAM destruídos; tickets são cancelados |
| Memória host não coerente | invalidate/flush correto antes/depois da materialização |
| Página com múltiplas authorities | watcher identifica owner/range/generation |
| Telemetria perdida | reason de trace gap e nenhuma decisão considerada comprovada |

## 11. Métricas e gates quantitativos

### 11.1 Correção

- zero loading hang;
- zero corrupção visual nas cenas já usadas;
- zero mismatch de bytes nas materializações validadas;
- zero stale label callback;
- zero guest upload proveniente de RAM stale;
- zero crescimento ilimitado de candidates, authorities, pins ou signals;
- 100% das decisões promovidas com correlação candidate/scope/version/consumer completa.

### 11.2 Causalidade da telemetria

- trace gaps e overwrites explicitamente contabilizados;
- primeiro consumidor conhecido para cada candidate concluído, ou terminal `session_end_unknown`;
- reason mask não-zero para todo fallback que não seja demanda CPU explícita;
- diferença entre decisão contrafactual e execução real registrada;
- sync requirement mask e classe de evitabilidade para toda decisão;
- `cause_id` presente em toda barrier, scope break, copy, flush, submit e wait atribuível;
- efeitos compartilhados identificados sem dupla contagem;
- tempo GPU inclusivo, exclusivo, compartilhado e não atribuído fechando no total por união de
  intervalos, dentro da precisão das queries;
- timestamp queries sem host wait introduzido pela coleta;
- queries vistas, disponíveis, perdidas, atrasadas e descartadas contabilizadas.

### 11.3 Performance

Comparar builds normais, mesma cena/save/config/driver, registrando média, p50, p95 e p99:

- frame time e frames abaixo de 60 fps;
- tempo ativo/bloqueado do GCP;
- host waits por motivo e duração;
- `WaitProgress`, submits e command buffer rollovers por frame;
- `EndRendering` por `ScopeBreakReason` e classe de evitabilidade;
- barriers por origem, stage/access/range, requirement e classe de evitabilidade;
- copies, bytes, staging e pins;
- materializações CPU e bytes realmente materializados;
- readbacks, barriers, scope breaks, flushes e submits separados em `ProvenRequired`,
  `ProvenEliminable`, `ConservativeFallback` e `UnknownDueToTraceGap`;
- GPU busy span, gaps entre command buffers e duração dos rendering scopes;
- tempo GPU por blocos de draw/dispatch, copy/tile/resolve, dependency-delay intervals e present;
- top pass/pipeline hashes por tempo inclusivo/exclusivo e por frame lento;
- host/guest draw, dispatch, pass e barrier ratios;
- shader invocations, primitives e transfer bytes por pipeline/pass quando amostrados;
- scaling de frame/pass em 0,5x/1x/2x de resolução;
- first-consumer distribution e authorities descartadas sem host use;
- custo exclusivo/compartilhado por fallback reason, não apenas eventos por frame.

O relatório deve apresentar, por efeito, `total`, `proven_required`, `conservative`,
`proven_eliminable` e `unknown`. O objetivo imediato é reduzir `conservative` com prova; um contador
menor sem mudança de classe causal não satisfaz o gate.

**GO:** correção preservada, cadeia causal completa, melhora reproduzível de frame pacing no build
normal e redução correspondente do custo GPU/GCP identificado. Para a cena GPU-limited, o trabalho
dominante precisa estar atribuído a passes/pipelines ou dependências concretas, não apenas ao uso
global reportado pelo driver.

**STOP/ROLLBACK:** corrupção, deadlock, loading hang, mismatch de versão/alias, stale callback,
piora de p95/p99, crescimento de memória/pins ou ganho existente apenas sob instrumentação.

## 12. Mapa de implementação por arquivo

### `src/video_core/texture_cache/texture_cache.h/.cpp`

- unificar metadados de candidate em `PendingImageDownload`;
- remover dependência funcional do singleton;
- classificar antes de `EndRendering`;
- registrar copy/scope-break/cause IDs do caminho real e contrafactual;
- preparar e commitar lotes transacionalmente;
- representar copy regions, subresources e alias epochs;
- manter o legado como fallback integral.

### `src/video_core/amdgpu/liverpool.h/.cpp`

- construir `CompletionScope` para EOS/EOP/ReleaseMem e sequências associadas;
- correlacionar SurfaceSync/AcquireMem/WaitRegMem;
- registrar intenção de hazard guest sem emitir automaticamente a dependência Vulkan;
- separar data action de logical signal action;
- remover promoção por assinatura do handler;
- manter semântica de GDS, IRQ e labels independente de readbacks.

### `src/video_core/gpu_authority_tracker.h/.cpp`

- retirar hardcodes de título e 512 bytes da lógica estrutural;
- evoluir para authority interval/version/generation;
- manter pending hazards por UID/epoch/alias/range até o consumidor real;
- produzir `SyncRequirementMask` e resolução producer->consumer;
- suportar múltiplos sinais e consumidores;
- materializar subranges e preservar versões anteriores;
- substituir gradualmente estruturas pointer-heavy por pool/index compacto.

### `src/video_core/buffer_cache/buffer_cache.h/.cpp`

- tornar image-to-buffer GPU o consumidor genérico de authority/shadow;
- integrar pending ranges com epochs, não apenas endereço;
- evitar staging guest quando houver representação GPU atual;
- garantir barriers e pins para tiled/linear conforme capacidade real.

### `src/video_core/texture_cache/image.h/.cpp`

- expor UID/epoch/subresource e transições ao ledger;
- registrar primeiro consumidor e dependência Vulkan aplicada;
- fornecer stage/access/layout/queue do producer e do consumidor ao resolver hazards;
- validar que a imagem original ainda pode servir como direct authority.

### `src/video_core/page_manager.h/.cpp`

- watchers por owner/range/generation;
- distinguir data read, label read, write, unmap e conflito de página;
- evitar que uma página observada force materialização de authorities não relacionadas.
- tornar `HasReadWatcher` coerente com `ApplyPageDelta` antes de usar o resultado como evidence bit.

### `src/video_core/renderer_vulkan/vk_scheduler.h/.cpp`

- tickets físicos e logical signals pendentes;
- motivos precisos para submit/progress;
- centralizar `EndRendering(ScopeBreakReason, cause_id)`;
- query-pool ring para timestamps e pipeline statistics amostrados, sem host wait;
- `VK_EXT_debug_utils` labels com IDs/hashes usados na telemetria/Nsight;
- ligar barriers, scope breaks, command buffers e submits aos `cause_id` participantes;
- batching somente após o ledger provar ausência de observação intermediária.

### `src/video_core/renderer_vulkan/vk_instance.h/.cpp`

- consultar e expor timestamp/pipeline-statistics/calibrated-timestamp capabilities;
- habilitar `VK_EXT_calibrated_timestamps` para telemetria quando suportada, independentemente de
  `TRACY_GPU_ENABLED`;
- consultar/habilitar `VK_KHR_dynamic_rendering_local_read` somente no experimento correspondente e
  expor seus limites ao resolvedor de hazards;
- registrar feature/property baseline no header do dump.

### `src/video_core/renderer_vulkan/vk_pipeline_cache.h/.cpp`

- manter hash estável de pipeline/pass nos brackets amostrados;
- permitir captura de `VK_KHR_pipeline_executable_properties` somente no modo diagnóstico;
- nunca habilitar capture flags estáticos na build normal.

### `src/common/performance_telemetry.h/.cpp`

- novo schema candidate/scope/decision/hazard/scope-break/effect/consumer/terminal;
- requirement mask, reason mask, evidence bits, evitabilidade e attribution confidence;
- agregação de intervalos inclusivos/exclusivos/compartilhados sem dupla contagem;
- ring health por stream;
- retirar taxonomia presa a compute/3K/512;
- writer assíncrono e dados frios deduplicados.

## 13. Ordem prática de trabalho

- [ ] Criar checkpoint e reproduzir normal/instrumentado.
- [ ] Registrar capabilities e implementar timestamp query ring sem host wait.
- [ ] Medir baseline/overhead e a cena GPU-heavy em 0,5x/1x/2x com cache aquecido.
- [ ] Desenhar structs fixed-size, requirement mask e schema causal relacional.
- [ ] Transformar todas as schedules em candidates observe-only.
- [ ] Decodificar CompletionScopes com base no SDK.
- [ ] Instrumentar pending hazards, first consumer, scope breaks, effects e terminal state.
- [ ] Integrar pipeline statistics amostradas e razões guest->host.
- [ ] Gerar dump e produzir matriz de requirements/reasons/evitabilidade/custo causal.
- [ ] Explicar causalmente o fast path atual e a cena GPU-limited; usar A/B mínimo se necessário.
- [ ] Generalizar GpuShadow para classes comprovadas.
- [ ] Validar loading, bytes, aliases, CPU reads e performance.
- [ ] Implementar authority interval/version e routing por consumidor.
- [ ] Resolver hazards no consumidor e reter rendering scopes quando comprovado.
- [ ] Generalizar logical signals/waits/IRQs.
- [ ] Implementar batching residual de copies/barriers/submits.
- [ ] Cruzar top pipeline hashes com executable stats/Nsight e testar extensões somente se as
  métricas justificarem.
- [ ] Consolidar estruturas DOP e remover dívida experimental.
- [ ] Fazer validação final normal/instrumentada e revisão de diff.

## 14. Não objetivos

- criar regras específicas para cada pacote, endereço, tamanho, formato ou título;
- inferir segurança somente porque não houve CPU read em uma captura;
- remover `finish()`, wait, barrier ou readback sem representação substituta;
- considerar label/IRQ como prova de consumo CPU dos dados;
- aceitar direct authority quando o consumidor é desconhecido;
- traduzir todo AcquireMem/EventWrite/SurfaceSync imediatamente para barrier ampla ou
  `EndRendering` sem consultar o consumidor;
- gravar uma barrier dentro de dynamic rendering sem satisfazer features e VUIDs aplicáveis;
- chamar um intervalo ao redor da barrier de “custo da barrier” sem provar a cadeia causal;
- somar intervalos GPU sobrepostos ou replicar integralmente custo compartilhado por reason;
- interpretar uso global de GPU, comparação de FLOPS PS4/RTX ou FPS instrumentado como diagnóstico
  suficiente;
- depender de `VK_KHR_performance_query` numa máquina cujo driver não a expõe;
- migrar para `vkQueueSubmit2` esperando ganho automático;
- usar extensão Vulkan para mascarar um hazard model incompleto;
- fazer uma reescrita grande antes do observe-only explicar o caso funcional;
- otimizar a build instrumentada como proxy do resultado final.

Depois do profiler básico, o primeiro entregável funcional deste plano não é um fast path maior. É
um classificador semântico observe-only capaz de responder, para **cada** readback legado, por que
ele existe, qual versão protege, qual pacote o ordena, quem consome os bytes e qual ação mais barata
preservaria exatamente a mesma observação. Só depois dessas respostas o mecanismo deixa de ser um
hack funcional e passa a ser uma implementação geral.

## 15. Referências normativas para a implementação

- [Vulkan — Synchronization and Cache Control](https://docs.vulkan.org/spec/latest/chapters/synchronization.html)
- [Vulkan — Queries](https://docs.vulkan.org/spec/latest/chapters/queries.html)
- [Vulkan — Pipelines and executable capture flags](https://docs.vulkan.org/spec/latest/chapters/pipelines.html)
- [VK_KHR_pipeline_executable_properties](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_pipeline_executable_properties.html)
- [NVIDIA Nsight Graphics — GPU Trace Overview](https://docs.nvidia.com/nsight-graphics/UserGuide/gpu-trace-overview.html)
- [NVIDIA Nsight Graphics — GPU Trace UI and markers](https://docs.nvidia.com/nsight-graphics/UserGuide/gpu-trace-ui.html)

O contrato guest continua sendo derivado prioritariamente dos headers, fontes e samples do SDK PS4
4.50 listados na Seção 3; as referências Vulkan definem apenas como realizar esse contrato no host.
