# Test serveur (logs)


import asyncio
from poke_env.player import RandomPlayer
from poke_env import LocalhostServerConfiguration

async def main():
    # Deux bots qui jouent l'un contre l'autre
    bot1 = RandomPlayer(
        battle_format="gen1randombattle",
        server_configuration=LocalhostServerConfiguration,
    )
    bot2 = RandomPlayer(
        battle_format="gen1randombattle",
        server_configuration=LocalhostServerConfiguration,
    )

    # bot1 challenge bot2, ils jouent 10 matchs
    await bot1.battle_against(bot2, n_battles=10)

    print(f"Bot1 wins: {bot1.n_won_battles} / 10")

asyncio.run(main())
