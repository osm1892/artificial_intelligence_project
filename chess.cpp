#include <iostream>
#include <vector>

using namespace std;

typedef pair<int, int> ip;
typedef pair<int, ip> iip;

class Game;

class Node;

class ChessGame;

int AlphaBetaSearch(Game *game, Node *state);

iip MaxValue(Game *game, Node *state, int alpha, int beta);

iip MinValue(Game *game, Node *state, int alpha, int beta);;
const int dy[8] = {-2, -2, -1, -1, 2, 2, 1, 1};
const int dx[8] = {-1, 1, -2, 2, -1, 1, -2, 2};

class Game
{
public:
	virtual Node *getInitial() = 0;

	virtual vector<ip> actions(Node *state) = 0;

	virtual Node result(Node *state, ip *move, int turn) = 0;

	bool isTerminal(Node *state)
	{
		return actions(state).empty();
	}

	virtual int utility(Node *state) = 0;
};

class Node
{
public:
	int size;
	int y;
	int x;
	vector<vector<int> > board;
	int point;

	Node(int size, int y, int x, vector<vector<int> > *board)
	{
		this->size = size;
		this->y = y;
		this->x = x;
		this->board = *board;
		this->point = 0;
	}

	explicit Node(Node *node)
	{
		size = node->size;
		y = node->y;
		x = node->x;
		board = node->board;
		point = node->point;
	}

	int utility() const
	{
		return point;
	}

	bool isValidPos(int y, int x)
	{
		if (y < 0 || size <= y)
		{
			return false;
		}
		if (x < 0 || size <= x)
		{
			return false;
		}
		if (board[y][x] == -1)
		{
			return false;
		}
		return true;
	}
};

class ChessGame : public Game
{
public:
	Node initial;

	ChessGame(int size, int y, int x, vector<vector<int> > *board) : initial(size, y, x, board)
	{

	}

	Node *getInitial()
	{
		return &initial;
	}

	vector<ip> actions(Node *state)
	{
		vector<ip> result;

		for (int i = 0; i < 8; i++)
		{
			int ny = state->y + dy[i];
			int nx = state->x + dx[i];

			if (!state->isValidPos(ny, nx))
			{
				continue;
			}

			result.push_back(make_pair(ny, nx));
		}

		return result;
	}

	Node result(Node *state, ip *move, int turn)
	{
		Node nextState = *state;
		int y = move->first;
		int x = move->second;

		if (turn == 0)
		{
			nextState.point += nextState.board[y][x];
		} else
		{
			nextState.point -= nextState.board[y][x];
		}
		nextState.board[y][x] = -1;
		return nextState;
	}

	int utility(Node *state)
	{
		return state->utility();
	}
};

int AlphaBetaSearch(Game *game, Node *state)
{
	int ans = -1e9;
	int size = state->size;

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			state->y = i;
			state->x = j;
			ans = max(ans, MaxValue(game, state, -1e9, 1e9).first);
		}
	}

	return ans;
}

iip MaxValue(Game *game, Node *state, int alpha, int beta)
{
	if (game->isTerminal(state))
	{
		return make_pair(game->utility(state), ip());
	}

	int v = -1e9;
	ip move = ip();

	vector<ip> actions = game->actions(state);

	for (int i = 0; i < actions.size(); i++)
	{
		Node result = game->result(state, &actions[i], 0);
		int v2 = MinValue(game, &result, alpha, beta).first;

		if (v2 > v)
		{
			v = v2;
			move = actions[i];
			alpha = max(alpha, v);
		}

		if (v >= beta)
		{
			return make_pair(v, move);
		}
	}

	return make_pair(v, move);
}

iip MinValue(Game *game, Node *state, int alpha, int beta)
{
	if (game->isTerminal(state))
	{
		return make_pair(game->utility(state), ip());
	}

	int v = 1e9;
	ip move = ip();

	vector<ip> actions = game->actions(state);

	for (int i = 0; i < actions.size(); i++)
	{
		Node result = game->result(state, &actions[i], 1);
		int v2 = MaxValue(game, &result, alpha, beta).first;

		if (v2 < v)
		{
			v = v2;
			move = actions[i];
			beta = min(beta, v);
		}

		if (v <= alpha)
		{
			return make_pair(v, move);
		}
	}

	return make_pair(v, move);
}

int main()
{
	int size = 0;

	cout << "체스판의 크기를 입력해주세요: ";

	cin >> size;
	cin.ignore();

	vector<vector<int> > board(size, vector<int>(size));

	cout << "체스판에 넣을 점수를 입력해주세요" << endl;

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			cin >> board[i][j];
			cin.ignore();
		}
	}

	ChessGame game(size, -1, -1, &board);

	int ans = AlphaBetaSearch(&game, &game.initial);

	cout << "플레이어 A와 B의 최대 점수차이는 " << ans << " 입니다." << endl;
}