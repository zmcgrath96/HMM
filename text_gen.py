import sys

def text_gen(args):
	try: 
		if args[0] == 'w':
			num_words = args[1]
		elif args[0] == 'a': 
			sentence =  args[1:]
		else:
			print('Incorrect input format')
	                print('Pass [w] [<int>] as arguments to generate <int> number o$
        	        print('Pass [a] [<string>] as arguments to autocomplete line')

	except Exception:
		print('Incorrect input format')
		print('Pass [w] [<int>] as arguments to generate <int> number of words')
		print('Pass [a] [<string>] as arguments to autocomplete line')


if __name__ == '__main__':
	text_gen(sys.argv[1:])
