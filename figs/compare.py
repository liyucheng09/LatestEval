def compare_strings(str_list1, str_list2, benchmarks, prompts):
    from difflib import SequenceMatcher

    all_result = []
    for str1, str2, title, prompt in zip(str_list1, str_list2, benchmarks, prompts):
        result = []
        result.append('<div class="text-box">')
        result.append(f'<div class="title">{title}</div>')
        result.append(f'<div class="prompt">{prompt}</div>')
        s = SequenceMatcher(None, str1, str2)
        
        for opcode, a0, a1, b0, b1 in s.get_opcodes():
            if opcode == 'equal':
                result.append(f'<span class="difference">{str1[a0:a1]}</span>')
            else:
                result.append(str1[a0:a1])
        result.append('</div>')
        all_result.append(''.join(result))
            
    return ''.join(all_result)

# Test the function
prompts = [
    """Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in""",
    """Persian (/ˈpɜːrʒən, -ʃən/), also known by its endonym Farsi (فارسی fārsi (fɒːɾˈsiː) ( listen)), is one of the Western Iranian languages within the Indo-Iranian branch of the Indo-European language family. It is primarily""",
    """The number of Malayalam speakers in Lakshadweep is 51,100, which is only 0.15% of the total number, but is as much as about 84% of the population of Lakshadweep. In all, Malayalis made up 3.22% of the total Indian population in 2001. Of the total 33,066,""",
    """The British have a love-hate relationship with the NHS. According to researchers at the King's Fund, the public gave the NHS its worst rating since records began 40 years ago. Just 29% said they were satisfied with the NHS in 2022. And yet we still love it. A whopping 90% of""",
]
predictions = [
    """ the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".""",
    """ spoken in Iran, Afghanistan (officially known as Dari since 1958),[3] and Tajikistan (officially known as Tajiki since the Soviet era),[4] and some other regions which historically were Persianate societies and considered part of Greater Iran. It is written right to left in the Persian alphabet, a modified variant of the Arabic script.""",
    """ 392 Malayalam speakers in India in 2001, 33,015,420 spoke the standard dialects, 19,643 spoke the Yerava dialect and 31,329 spoke non-standard regional variations like Eranadan. As per the 1991 census data, 28.85% of all Malayalam speakers in India spoke a second language and 19.64% of the total knew three or more languages.""",
    """ of Britons believe that the NHS is a crucial institution that should be preserved. The juxtaposition between dissatisfaction with the current state and overall reverence for the institution speaks volumes about the complex relationship the British public has with their healthcare system."""

]
references = [
    """ the late 1990s as lead singer of the R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Her debut solo album, Dangerously in Love (2003), debuted at number one on the US Billboard 200 chart and earned her five Grammy Awards, solidifying her as a solo artist as well. Throughout her career, Beyoncé has sold over 100 million records worldwide as a solo artist and a further 60 million records with Destiny's Child, making her one of the best-selling music artists of all time. She has won 23 Grammy Awards and is the most""",
    """ spoken in Iran, Afghanistan (officially known as Dari since 1958), and Tajikistan (officially known as Tajiki since the Soviet era), and some other regions which historically were Persianate societies and considered part of Greater Iran. It is written in the Persian alphabet, a modified variant of the Arabic script, which itself evolved from the Aramaic alphabet.""",
    """ 392 Malayalam speakers in India in 2001, 33,015,420 spoke the standard dialects, 19,643 spoke the Yerava dialect and 31,329 spoke non-standard regional variations like Eranadan. As per the 1991 census data, 28.85% of all Malayalam speakers in India spoke a second language and 19.64% of the total knew three or more languages.""",
    """of the public agrees the service should be free and available to everyone. But with more than seven million people on waiting lists, almost everyone knows someone who isn't getting the care they need. As the NHS approaches its 75th anniversary, politicians are falling over themselves to praise the service."""
]
benchmarks = ['squad', 'boolq', 'quac', 'LatestEval']
html_output = compare_strings(predictions, references, benchmarks, prompts)

with open('output.html', 'w') as f:
    f.write(f"""
    <html>
    <head>
        <style>
            .difference {{
                background: #a9d0f5;
            }}
            .text-box {{
                width: 300px;  /* Adjust as needed */
                height: 200px; /* Adjust as needed */
                border: 1px solid #000;
                padding: 10px;
                overflow: auto;
            }}
            .title {{
                font-size: 20px;
                font-weight: bold;
                margin-bottom: 10px;
            }}
            .container {{
                display: flex;
                gap: 20px; /* provides space between the boxes */
            }}
            .prompt {{
                font-size: 10px;
                font-style: italic;
                margin-bottom: 4px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            {html_output}
        </div>
    </body>
    </html>
    """)

print("HTML output saved to 'output.html'")
