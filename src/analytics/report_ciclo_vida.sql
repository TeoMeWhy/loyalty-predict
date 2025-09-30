SELECT dtRef,
       descLifeCycle,
       count(*) AS qtdeCliente

FROM life_cycle

WHERE descLifeCycle <> '05-ZUMBI'

group by dtRef, descLifeCycle
order by dtRef, descLifeCycle