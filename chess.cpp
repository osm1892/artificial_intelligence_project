#include <iostream>
#include <vector>

using namespace std;

typedef pair<int, int> ii;
typedef pair<int, ii> iii;

// 상태공간에 기반한 게임 클래스입니다.
class Game;

// 상태공간에서의 각 상태를 나타내는 노드 클래스입니다.
class Node;

// 하나의 나이트를 가지고 땅따먹기를 하는 체스게임의 축소 문제입니다.
class ChessGame;

// 알파-베타 탐색을 수행하는 함수입니다.
int AlphaBetaSearch(Game *game, Node *state);

// 알파-베타 탐색의 MaxValue 부분입니다.
iii MaxValue(Game *game, Node *state, int alpha, int beta);

// 알파-베타 탐색의 MinValue 부분입니다.
iii MinValue(Game *game, Node *state, int alpha, int beta);

// 나이트의 다음 이동 경로를 저장하는 변수입니다.
const int dy[8] = {-2, -2, -1, -1, 2, 2, 1, 1};
const int dx[8] = {-1, 1, -2, 2, -1, 1, -2, 2};

class Game
{
public:
	virtual void getInitial(Node *out) = 0;

	// 주어진 상태에서 허용 가능한 수 리스트를 반환
	virtual void actions(Node *state, vector<ii> *result) = 0;

	// 주어진 상태에서 수를 두었을 때의 결과 상태 반환
	virtual void result(Node *state, ii *move, int turn, Node *result) = 0;

	// 종료 상태 여부를 반환
	bool isTerminal(Node *state)
	{
		vector<ii> action_list;
		actions(state, &action_list);
		return action_list.empty();
	}

	// 종료 상태에서 게임이 종료되었을 때의 효용값 반환
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
		this->board.assign(board->begin(), board->end());
		this->point = 0;
	}

	explicit Node(Node *node)
	{
		size = node->size;
		y = node->y;
		x = node->x;
		board.assign(node->board.begin(), node->board.end());
		point = node->point;
	}

	Node(Node const &node) : size(node.size), y(node.y), x(node.x), point(node.point)
	{
		board.assign(node.board.begin(), node.board.end());
	}

	Node()
	{
		size = 0;
		y = 0;
		x = 0;
		point = 0;
	}

	void init(Node *node)
	{
		size = node->size;
		y = node->y;
		x = node->x;
		board.assign(node->board.begin(), node->board.end());
		point = node->point;
	}

	// 현 상태의 효용값 반환
	int utility() const
	{
		return point;
	}

	// 주어진 좌표에 말을 놓을 수 있는지 체크
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

	explicit ChessGame(ChessGame *other) : initial(&(other->initial))
	{

	}

	void getInitial(Node *out)
	{
		out = &initial;
	}

	void actions(Node *state, vector<ii> *result)
	{
		result->clear();

		// 8개의 이동 경로를 순회하면서 가능한 이동 경로를 저장합니다.
		for (int i = 0; i < 8; i++)
		{
			int ny = state->y + dy[i];
			int nx = state->x + dx[i];

			if (!state->isValidPos(ny, nx))
			{
				continue;
			}

			result->push_back(make_pair(ny, nx));
		}
	}

	void result(Node *state, ii *move, int turn, Node *result)
	{
		Node nextState(state);
		int y = move->first;
		int x = move->second;

		// turn이 0인 경우 Max, 1인 경우 Min을 의미합니다.
		if (turn == 0)
		{
			nextState.point += nextState.board[y][x];
		} else
		{
			nextState.point -= nextState.board[y][x];
		}
		nextState.board[y][x] = -1;

		// 새로운 상태를 result 변수에 복사합니다.
		result->init(&nextState);
	}

	int utility(Node *state)
	{
		return state->utility();
	}
};

int AlphaBetaSearch(Game *game, Node *state)
{
	// Max가 게임을 시작하는 것으로 알파-베타 탐색을 시작합니다.
	return MaxValue(game, state, -1e9, 1e9).first;
}

iii MaxValue(Game *game, Node *state, int alpha, int beta)
{
	if (game->isTerminal(state))
	{
		return make_pair(game->utility(state), ii());
	}

	int v = -1e9;
	ii move = ii();

	vector<ii> actions;
	game->actions(state, &actions);

	for (int i = 0; i < actions.size(); i++)
	{
		Node result;
		game->result(state, &actions[i], 0, &result);
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

iii MinValue(Game *game, Node *state, int alpha, int beta)
{
	if (game->isTerminal(state))
	{
		return make_pair(game->utility(state), ii());
	}

	int v = 1e9;
	ii move = ii();

	vector<ii> actions;
	game->actions(state, &actions);

	for (int i = 0; i < actions.size(); i++)
	{
		Node result;
		game->result(state, &actions[i], 1, &result);
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

	ChessGame game(size, 0, 0, &board);

	int ans = AlphaBetaSearch(&game, &game.initial);

	cout << "플레이어 A와 B의 최대 점수차이는 " << ans << " 입니다." << endl;
}