# Hangman game
import random

WORDLIST_FILENAME = "words.txt"

def loadWords():
	"""
	Returns a list of valid words. Words are strings of lowercase letters.
	Depending on the size of the word list, this function may
	take a while to finish.
	"""
	print("Loading word list from file...")
	# inFile: file
	inFile = open(WORDLIST_FILENAME, 'r')
	# line: string
	line = inFile.readline()
	# wordlist: list of strings
	wordlist = line.split()
	print("  ", len(wordlist), "words loaded.")
	return wordlist

def chooseWord(wordlist):
	"""
	wordlist (list): list of words (strings)

	Returns a word from wordlist at random
	"""
	return random.choice(wordlist)


def isWordGuessed(secretWord, lettersGuessed):
	''' 
	secretWord: string, the word the user is guessing
	lettersGuessed: list, what letters have been guessed so far 
	returns: boolean, True if all the letters of secretWord are in lettersGuessed;
	False otherwise
	''' 
	# FILL IN YOUR CODE HERE...
	for char in lettersGuessed:
		if char in secretWord:
			secretWord = secretWord.replace(char,"")
	if secretWord == "": 
		return True
	else:
		return False


def getGuessedWord(secretWord, lettersGuessed):
	''' 
	secretWord: string, the word the user is guessing
	lettersGuessed: list, what letters have been guessed so far 
	returns: string, comprised of letters and underscores that represents
	what letters in secretWord have been guessed so far.
	''' 
	guessWord = ""
	for i in range(len(secretWord)):
		if secretWord[i] in lettersGuessed:
			guessWord = guessWord + secretWord[i]
		elif not secretWord[i] in lettersGuessed and i < len(secretWord) - 1:
			guessWord = guessWord + "_ "
		else:
			guessWord = guessWord + "_" 
	return guessWord



def getAvailableLetters(lettersGuessed):
	'''
	lettersGuessed: list, what letters have been guessed so far
	returns: string, comprised of letters that represents what letters have not
	yet been guessed.
	'''
	available = "abcdefghijklmnopqrstuvwxyz"
	for char in lettersGuessed:
		if char in available:
			available = available.replace(char, "")
	return available
 

def hangman(secretWord):
	'''
	secretWord: string, the secret word to guess.

	Starts up an interactive game of Hangman.

	* At the start of the game, let the user know how many
	letters the secretWord contains.

	* Ask the user to supply one guess (i.e. letter) per round.

	* The user should receive feedback immediately after each guess
	about whether their guess appears in the computers word.

	* After each round, you should also display to the user the
	partially guessed word so far, as well as letters that the
	user has not yet guessed.

	Follows the other limitations detailed in the problem write-up.
	'''
	lettersGuessed = ""
	numGuesses = 8
	print("Welcome to the game Hangman!")
	print("I am thinking of a word that is", len(secretWord), "letters long.")
	print("-------------")
	while True:
		if numGuesses > 0:
			print("You have", numGuesses, "guesses left.")
			print("Available letters: ", getAvailableLetters(lettersGuessed))
			guess = input("Please guess a letter: ").lower()
			if guess in lettersGuessed:
				print("Oops! You've already guessed that letter:", getGuessedWord(secretWord, lettersGuessed))
				print("-------------")	
			elif guess in secretWord:
				lettersGuessed = lettersGuessed + guess
				print("Good guess:", getGuessedWord(secretWord, lettersGuessed))
				print("-------------")
				if isWordGuessed(secretWord, lettersGuessed):
					print("Congratulations, you won!")
					break
			else:
				print("Oops! That letter is not in my word:", getGuessedWord(secretWord, lettersGuessed))
				print("-------------")
				numGuesses -= 1
				lettersGuessed = lettersGuessed + guess
		else:
			print("Sorry, you ran out of guesses. The word was " + secretWord + ".")
			break


# Load the list of words into the variable wordlist
# so that it can be accessed from anywhere in the program
wordlist = loadWords()
secretWord = chooseWord(wordlist).lower()
hangman(secretWord)
