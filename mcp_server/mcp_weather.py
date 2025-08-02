from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather",host = "0.0.0.0",port = 9000)

@mcp.tool()
async def get_weather(location: str) -> str:
    """
    获取任意地方的天气
    输入：
        location: 地点
    输出：
        天气
    """
    return f"{location}是晴天"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")