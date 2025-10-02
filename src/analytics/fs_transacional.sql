WITH tb_transacao AS (

    SELECT *,
           substr(DtCriacao,0,11) AS dtDia
    FROM transacoes
    WHERE dtCriacao < '2025-10-01'

),

tb_agg_transacao AS (

    SELECT IdCliente,

            count(DISTINCT dtDia) AS qtdeAtivacaoVida,
            count(DISTINCT CASE WHEN dtDia >= date('2025-10-01', '-7 day') THEN dtDia END) AS qtdeAtivacaoD7,
            count(DISTINCT CASE WHEN dtDia >= date('2025-10-01', '-14 day') THEN dtDia END) AS qtdeAtivacaoD14,
            count(DISTINCT CASE WHEN dtDia >= date('2025-10-01', '-28 day') THEN dtDia END) AS qtdeAtivacaoD28,
            count(DISTINCT CASE WHEN dtDia >= date('2025-10-01', '-56 day') THEN dtDia END) AS qtdeAtivacaoD56,

            count(DISTINCT IdTransacao) AS qtdeTransacaoVida,
            count(DISTINCT CASE WHEN dtDia >= date('2025-10-01', '-7 day') THEN IdTransacao END) AS qtdeTransacaoD7,
            count(DISTINCT CASE WHEN dtDia >= date('2025-10-01', '-14 day') THEN IdTransacao END) AS qtdeTransacaoD14,
            count(DISTINCT CASE WHEN dtDia >= date('2025-10-01', '-28 day') THEN IdTransacao END) AS qtdeTransacaoD28,
            count(DISTINCT CASE WHEN dtDia >= date('2025-10-01', '-56 day') THEN IdTransacao END) AS qtdeTransacaoD56,

            sum(qtdePontos) AS saldoVida,
            sum(CASE WHEN dtDia >= date('2025-10-01', '-7 day') THEN qtdePontos ELSE 0 END) AS saldoD7,
            sum(CASE WHEN dtDia >= date('2025-10-01', '-14 day') THEN qtdePontos ELSE 0 END) AS saldoD14,
            sum(CASE WHEN dtDia >= date('2025-10-01', '-28 day') THEN qtdePontos ELSE 0 END) AS saldoD28,
            sum(CASE WHEN dtDia >= date('2025-10-01', '-56 day') THEN qtdePontos ELSE 0 END) AS saldoD56,

            sum(CASE WHEN qtdePontos > 0 THEN qtdePontos ELSE 0 END ) AS qtdePontosPosVida,
            sum(CASE WHEN dtDia >= date('2025-10-01', '-7 day') AND qtdePontos > 0 THEN qtdePontos ELSE 0 END) AS qtdePontosPosD7,
            sum(CASE WHEN dtDia >= date('2025-10-01', '-14 day') AND qtdePontos > 0 THEN qtdePontos ELSE 0 END) AS qtdePontosPosD14,
            sum(CASE WHEN dtDia >= date('2025-10-01', '-28 day') AND qtdePontos > 0 THEN qtdePontos ELSE 0 END) AS qtdePontosPosD28,
            sum(CASE WHEN dtDia >= date('2025-10-01', '-56 day') AND qtdePontos > 0 THEN qtdePontos ELSE 0 END) AS qtdePontosPosD56,

            sum(CASE WHEN qtdePontos < 0 THEN qtdePontos ELSE 0 END ) AS qtdePontosNegVida,
            sum(CASE WHEN dtDia >= date('2025-10-01', '-7 day') AND qtdePontos < 0 THEN qtdePontos ELSE 0 END) AS qtdePontosNegD7,
            sum(CASE WHEN dtDia >= date('2025-10-01', '-14 day') AND qtdePontos < 0 THEN qtdePontos ELSE 0 END) AS qtdePontosNegD14,
            sum(CASE WHEN dtDia >= date('2025-10-01', '-28 day') AND qtdePontos < 0 THEN qtdePontos ELSE 0 END) AS qtdePontosNegD28,
            sum(CASE WHEN dtDia >= date('2025-10-01', '-56 day') AND qtdePontos < 0 THEN qtdePontos ELSE 0 END) AS qtdePontosNegD56

    FROM tb_transacao
    GROUP BY IdCliente

),

tb_agg_calc AS (

    SELECT 
            *,
            COALESCE(1. * qtdeTransacaoVida / qtdeAtivacaoVida,0) AS QtdeTransacaoDiaVida,
            COALESCE(1. * qtdeTransacaoD7 / qtdeAtivacaoD7,0) AS QtdeTransacaoDiaD7,
            COALESCE(1. * qtdeTransacaoD14 / qtdeAtivacaoD14,0) AS QtdeTransacaoDiaD14,
            COALESCE(1. * qtdeTransacaoD28 / qtdeAtivacaoD28,0) AS QtdeTransacaoDiaD28,
            COALESCE(1. * qtdeTransacaoD56 / qtdeAtivacaoD56,0) AS QtdeTransacaoDiaD56,
            COALESCE(1. * qtdeAtivacaoD28 / 28, 0) AS pctAtivacaoMAU

    FROM tb_agg_transacao

),

tb_horas_dia AS (

    SELECT IdCliente,
           dtDia,
           24 * (max(julianday(DtCriacao)) - min(julianday(DtCriacao))) AS duracao

    FROM tb_transacao
    GROUP BY IdCliente, dtDia

),

tb_hora_cliente AS (

    SELECT IdCliente,
           sum(duracao) AS qtdeHorasVida,
           sum(CASE WHEN dtDia >= date('2025-10-01', '-7 day') THEN duracao ELSE 0 END) AS qtdeHorasD7,
           sum(CASE WHEN dtDia >= date('2025-10-01', '-14 day') THEN duracao ELSE 0 END) AS qtdeHorasD14,
           sum(CASE WHEN dtDia >= date('2025-10-01', '-28 day') THEN duracao ELSE 0 END) AS qtdeHorasD28,
           sum(CASE WHEN dtDia >= date('2025-10-01', '-56 day') THEN duracao ELSE 0 END) AS qtdeHorasD56

    FROM tb_horas_dia
    GROUP BY IdCliente
),

tb_lag_dia AS (

    SELECT idCliente,
           dtDia,
           LAG(dtDia) OVER (PARTITION BY idCliente order by dtDia) AS lagDia

    FROM tb_horas_dia

),

tb_intervalo_dias AS (

    SELECT IdCliente,
           avg(julianday(dtDia) - julianday(lagDia)) AS avgIntervaloDiasVida,
           avg(CASE WHEN dtDia >= date('2025-10-01', '-28 day') THEN julianday(dtDia) - julianday(lagDia) END) AS avgIntervaloDiasD28

    FROM tb_lag_dia
    GROUP BY idCliente

)

SELECT t1.*,
       t2.qtdeHorasVida,
       t2.qtdeHorasD7,
       t2.qtdeHorasD14,
       t2.qtdeHorasD28,
       t2.qtdeHorasD56,
       t3.avgIntervaloDiasVida,
       t3.avgIntervaloDiasD28

FROM tb_agg_calc AS t1

LEFT JOIN tb_hora_cliente AS t2
ON t1.IdCliente = t2.IdCliente

LEFT JOIN tb_intervalo_dias AS t3
ON t1.IdCliente = t3.IdCliente