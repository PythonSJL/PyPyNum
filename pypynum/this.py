"""
The Zen of PyPyNum
"""

Zen = "\x9a¤¢\xa0¢°¼§¢°éâñåâòõÖ°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°\x9a\x9a±äù°õóñâòýõ°ã·äõü½äàõóþÿó°äåÿôþñäã°ñ°ãù°þ÷ùãõô°âñüåôÿÝ\x9a¾÷þùãùýÿâà°õò°äø÷ùý°äù°¼ôâñçâÿöäø÷ùñâäã°ãù°þÿùäñäþõýõüàýù°õøä°öÙ\x9a¾äù°âõôùãþÿóõâ°¼ôþõøõâàýÿó°ÿä°÷þù÷þõüüñøó°ãù°þÿùäñäþõýõüàýù°õøä°öÙ\x9a¾÷þùøãåâ°ÿä°âÿùâõàåã°ãù°÷þùäùñç°ãõýùäõýÿã°¼âõæõçÿØ\x9a¾þÿùäñþùäãñâóÿâà°þñøä°âõääõò°ãù°éüäàýÿâà°÷þùäóÑ\x9a¾øóñÿâààñ°ãåÿùæòÿ½õþÿ°éüþÿ°õøä°¼éüüñõôù½õþÿ°õò°äãåý°õâõøÄ\x9a¾õäñüåóõàã°ÿä°õ÷âå°õøä°äãùãõâ°¼ããõþõå÷ñæ°öÿ°õóþõãõâà°õøä°þÙ\x9a¾ôõäåý°éüäùóùüàèõ°ããõüþÅ\x9a¾ôõóþõüùã°äÿþ°¼þõûÿàã°õò°äãåý°ãâÿââÕ\x9a¾éäùâåà°âõæÿ°ãøàýåùâä°þõäöÿ°éäùüñóùäóñâÀ\x9a¾ãýâÿþ°õøä°õôùââõæÿ°äÿþ°ôüåÿøã°ãõãñó°üñùóõàÃ\x9a¾äþåÿýñâñà°ãù°éäùüùòñôñõÂ\x9a¾ãõþÿ°ôõäñÿüò°âõæÿ°ãþùç°õôÿó°õãâñàÃ\x9a¾ãõùøóâñâõùø°ôõäãõþ°ãäñõò°õâåäóåâäã°äñüÖ\x9a¾ôõäñóùüàýÿóâõæÿ°þñøä°âõääõò°ãù°ôõäñóùäãùøàÿÃ\x9a¾þÿùäåüÿæþÿó°âõæÿ°ôõââõöõâà°ãù°ããõþôâñçâÿöäø÷ùñâäÃ\x9a¾éäùâåóãòÿ°ãàýåâä°éäùâñüÓ\x9a¾éûþåüó°ÿä°âÿùâõàåã°ãù°äþñ÷õüÕ\x9a\x9a¾þÿøäéÀ°þù°éüõâåà°þõääùâç°õ÷ñûóñà°øäñý°ñ°ãù°ãùøÄ\x9a\x9aùéñùÚ°þõøÃ°éò°¼ýåÞéÀéÀ°öÿ°þõÊ°õøÄ°°°°\x9a"
print("".join([chr(ord(_) ^ 144) for _ in Zen[::-1]]))
for _ in list(globals().keys()):
    if _ != "__builtins__":
        del globals()[_]
del _
