# Bot random pour jouer en Vs Gen1


import asyncio
from poke_env.player import RandomPlayer
from poke_env import LocalhostServerConfiguration

async def main():
    bot = RandomPlayer(
        battle_format="gen1randombattle",
        server_configuration=LocalhostServerConfiguration,
    )
    
    # Le bot attend un challenge humain
    await bot.accept_challenges(None, 1000)  # None = accepte n'importe qui, 1 = 1 match

asyncio.run(main())